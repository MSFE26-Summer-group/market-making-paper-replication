"""Tests for the RL sample model: data pipeline, environment, model."""

import numpy as np
import pandas as pd
import pytest
import torch

from paper_replication.rl.data import LOB_COLS, LOBDataset, build_dataset
from paper_replication.rl.env import EnvConfig, MarketMakingEnv, StepResult
from paper_replication.rl.model import ActorCritic, AttnLOB
from paper_replication.rl.train import (
    PPOConfig,
    as_policy,
    collect_rollout,
    compute_gae,
    evaluate,
    fixed_policy,
    ppo_update,
    random_policy,
)
from paper_replication.rl.train import test_starts as eval_starts


def make_dataset(n: int = 500, window: int = 10) -> LOBDataset:
    """Synthetic dataset: flat mid at 100, trades within +-2 bps."""
    rng = np.random.default_rng(1)
    mid = 100.0 + rng.normal(0, 0.001, n).cumsum()
    return LOBDataset(
        states=rng.normal(0, 1, (n, window, len(LOB_COLS))).astype(np.float32),
        mid=mid,
        trade_max=mid * (1 + 2e-4),
        trade_min=mid * (1 - 2e-4),
        imbalance=rng.uniform(-1, 1, n),
        spread=np.full(n, 0.01),
        size_mean=np.zeros(20),
        size_std=np.ones(20),
    )


class TestEnv:
    def test_reset_zeroes_position(self) -> None:
        env = MarketMakingEnv(make_dataset(), EnvConfig(episode_len=50))
        res = env.reset(start=0)
        assert isinstance(res, StepResult)
        assert env.inventory == 0 and env.cash == 0.0
        assert res.aux.shape == (3,)

    def test_tight_quotes_get_filled(self) -> None:
        env = MarketMakingEnv(make_dataset(), EnvConfig(episode_len=50))
        env.reset(start=0)
        res = env.step(np.array([0.0, -1.0]))  # tightest allowed spread
        # min half spread 0.5 bps < trade range 2 bps: both sides fill,
        # inventory nets to zero and we earn the spread.
        assert res.info["trading_pnl"] > 0
        assert env.inventory == 0

    def test_wide_quotes_never_fill(self) -> None:
        env = MarketMakingEnv(make_dataset(), EnvConfig(episode_len=50))
        env.reset(start=0)
        res = env.step(np.array([0.0, 1.0]))  # 10 bps half spread
        assert res.info["trading_pnl"] == 0.0
        assert env.inventory == 0

    def test_inventory_capped(self) -> None:
        cfg = EnvConfig(episode_len=300, max_inventory=3, max_bias_bps=50)
        env = MarketMakingEnv(make_dataset(), cfg)
        env.reset(start=0)
        for _ in range(100):
            # Quote far above mid: only the bid side can fill.
            env.step(np.array([1.0, -1.0]))
        assert abs(env.inventory) <= cfg.max_inventory

    def test_episode_terminates_and_liquidates(self) -> None:
        env = MarketMakingEnv(make_dataset(), EnvConfig(episode_len=20))
        res = env.reset(start=0)
        steps = 0
        while not res.done:
            res = env.step(np.array([0.0, 0.0]))
            steps += 1
        assert steps == 20
        assert env.inventory == 0


class TestModel:
    def test_encoder_output_shape(self) -> None:
        enc = AttnLOB(n_features=40, d_model=32)
        out = enc(torch.randn(8, 10, 40))
        assert out.shape == (8, 32)

    def test_actor_critic_shapes_and_bounds(self) -> None:
        model = ActorCritic(n_features=40, n_aux=3, d_model=32)
        dist, value = model(torch.randn(4, 10, 40), torch.randn(4, 3))
        assert dist.mean.shape == (4, 2)
        assert value.shape == (4,)
        assert torch.all(dist.mean.abs() <= 1.0)

    def test_act_returns_logp_and_value(self) -> None:
        model = ActorCritic(n_features=40, n_aux=3, d_model=32)
        action, logp, value = model.act(torch.randn(1, 10, 40), torch.randn(1, 3))
        assert action.shape == (1, 2)
        assert logp.shape == (1,)

    def test_side_aware_fills(self) -> None:
        ds = make_dataset()
        # Sellers only printed 1 bp below mid; buyers only 1 bp above.
        ds.sell_min = ds.mid * (1 - 1e-4)
        ds.buy_max = ds.mid * (1 + 1e-4)
        env = MarketMakingEnv(ds, EnvConfig(episode_len=50))
        env.reset(start=0)
        res = env.step(np.array([0.0, -1.0]))  # 0.5 bp half spread
        assert res.info["fills"] == 2.0  # both sides crossed
        env.reset(start=10)
        res = env.step(np.array([0.0, 0.0]))  # ~5 bp half spread
        assert res.info["fills"] == 0.0  # no side reaches the quotes


class TestDataPipeline:
    def test_build_dataset_shapes_and_normalization(self) -> None:
        n, window = 200, 10
        rng = np.random.default_rng(2)
        mid = 100.0 + rng.normal(0, 0.01, n).cumsum()
        cols: dict[str, np.ndarray] = {}
        for c in LOB_COLS:
            lvl = int(c.rsplit("_", 1)[1])
            sign = -1.0 if "bid" in c else 1.0
            if "price" in c:
                cols[c] = mid * (1 + sign * lvl * 1e-4)
            else:
                cols[c] = rng.lognormal(0, 1, n)
        df = pd.DataFrame(cols)
        df["mid_price"] = mid
        df["max_trade_price"] = mid * (1 + 1e-4)
        df["min_trade_price"] = mid * (1 - 1e-4)
        df["lob_imbalance"] = rng.uniform(-1, 1, n)
        df["best_bid"] = mid * (1 - 0.5e-4)
        df["best_ask"] = mid * (1 + 0.5e-4)

        ds = build_dataset(df, window=window, train_frac=0.7)
        assert ds.states.shape == (n - window + 1, window, len(LOB_COLS))
        assert ds.states.dtype == np.float32
        assert len(ds.mid) == n - window + 1
        # Level-1 bid sits 1 bp below mid -> price feature is about -1.
        assert ds.states[0, -1, 0] == pytest.approx(-1.0, abs=0.05)
        # Size features are z-scored: roughly centered on the train split.
        assert abs(ds.states[:, :, 1::2].mean()) < 0.5


class TestPPO:
    def test_rollout_and_update_run(self) -> None:
        torch.manual_seed(0)
        env = MarketMakingEnv(make_dataset(n=120, window=10), EnvConfig(episode_len=15))
        model = ActorCritic(n_features=40, n_aux=3, d_model=16)
        cfg = PPOConfig(rollout_episodes=2, epochs=1, minibatch=16)
        batch = collect_rollout(env, model, cfg)
        assert batch["states"].shape[0] == 30  # 2 episodes x 15 steps
        assert batch["adv"].shape == batch["logp"].shape
        loss = ppo_update(
            model, torch.optim.Adam(model.parameters(), lr=1e-3), batch, cfg
        )
        assert np.isfinite(loss)


class TestEvaluate:
    def test_metrics_for_benchmark_policies(self) -> None:
        ds = make_dataset(n=2000, window=10)
        starts = eval_starts(ds, episode_len=360)
        assert len(starts) >= 1
        for policy in (
            fixed_policy(-1.0),
            random_policy(np.random.default_rng(0)),
            as_policy(ds),
        ):
            m = evaluate(ds, policy, starts[:1])
            assert set(m) >= {
                "pnl_mean",
                "sharpe",
                "nd_pnl",
                "pnl_map",
                "fills_per_ep",
            }
            assert np.isfinite(m["pnl_mean"])

    def test_tight_fixed_policy_profits_on_flat_market(self) -> None:
        # Flat mid + symmetric 2 bp trade range: tight quotes earn spread.
        ds = make_dataset(n=1000, window=10)
        m = evaluate(ds, fixed_policy(-1.0), [0])
        assert m["pnl_mean"] > 0
        assert m["fills_per_ep"] > 100


class TestGAE:
    def test_advantage_shapes_and_terminal_reset(self) -> None:
        cfg = PPOConfig()
        rewards = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        values = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        dones = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        adv, ret = compute_gae(rewards, values, dones, cfg)
        assert adv.shape == rewards.shape
        # Terminal steps bootstrap from nothing: adv = r - v exactly.
        assert adv[1] == pytest.approx(1.0 - 0.5)
        assert adv[3] == pytest.approx(1.0 - 0.5)
        assert np.allclose(ret, adv + values)
