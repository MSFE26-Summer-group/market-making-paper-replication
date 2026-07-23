"""PPO training and evaluation harness for the market-making agent.

Train (tick-based fills, 500 updates), then evaluate on the test split:

    PYTHONPATH=src uv run --no-sync python -m paper_replication.rl.train \
        --updates 500 --fills tick --out results/run_tick

Evaluate benchmark strategies only (no training):

    PYTHONPATH=src uv run --no-sync python -m paper_replication.rl.train \
        --benchmarks-only --out results/benchmarks

The first 70% of snapshots are the training segment (episode starts are
sampled there); the last 30% are held out for evaluation. Metrics follow
the paper: episode PnL, Sharpe over episodes, PnLMAP (PnL per unit of
mean absolute position) and ND-PnL (PnL / average market spread).
"""

import argparse
import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from paper_replication.rl.data import (
    LOBDataset,
    attach_tick_fills,
    build_dataset,
    load_lob,
)
from paper_replication.rl.env import EnvConfig, MarketMakingEnv, StepResult
from paper_replication.rl.model import ActorCritic

TRAIN_FRAC = 0.7

Policy = Callable[[MarketMakingEnv, StepResult], np.ndarray]


@dataclass
class PPOConfig:
    rollout_episodes: int = 4
    epochs: int = 4
    minibatch: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, cfg: PPOConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation over concatenated episodes."""
    adv = np.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + cfg.gamma * next_value * nonterminal - values[t]
        last = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last
        adv[t] = last
    returns = adv + values[: len(adv)]
    return adv, returns


def collect_rollout(
    env: MarketMakingEnv, model: ActorCritic, cfg: PPOConfig
) -> dict[str, Tensor]:
    """Run episodes with the current policy and stack the transitions."""
    buf: dict[str, list[np.ndarray | float]] = {
        k: []
        for k in ("states", "aux", "actions", "logp", "values", "rewards", "dones")
    }
    ep_rewards: list[float] = []
    for _ in range(cfg.rollout_episodes):
        res = env.reset()
        ep_total = 0.0
        while not res.done:
            s = torch.from_numpy(res.state).unsqueeze(0)
            a = torch.from_numpy(res.aux).unsqueeze(0)
            action, logp, value = model.act(s, a)
            buf["states"].append(res.state)
            buf["aux"].append(res.aux)
            buf["actions"].append(action.numpy()[0])
            buf["logp"].append(float(logp))
            buf["values"].append(float(value))
            res = env.step(action.numpy()[0])
            buf["rewards"].append(res.reward)
            buf["dones"].append(float(res.done))
            ep_total += res.reward
        ep_rewards.append(ep_total)

    adv, returns = compute_gae(
        np.array(buf["rewards"], dtype=np.float32),
        np.array(buf["values"], dtype=np.float32),
        np.array(buf["dones"], dtype=np.float32),
        cfg,
    )
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    out = {
        "states": torch.from_numpy(np.array(buf["states"], dtype=np.float32)),
        "aux": torch.from_numpy(np.array(buf["aux"], dtype=np.float32)),
        "actions": torch.from_numpy(np.array(buf["actions"], dtype=np.float32)),
        "logp": torch.tensor(buf["logp"], dtype=torch.float32),
        "adv": torch.from_numpy(adv),
        "returns": torch.from_numpy(returns),
    }
    out["ep_reward_mean"] = torch.tensor(float(np.mean(ep_rewards)))
    return out


def ppo_update(
    model: ActorCritic,
    optim: torch.optim.Optimizer,
    batch: dict[str, Tensor],
    cfg: PPOConfig,
) -> float:
    """Clipped-objective PPO epochs over the rollout; returns last loss."""
    n = len(batch["logp"])
    loss = torch.tensor(0.0)
    for _ in range(cfg.epochs):
        for idx in torch.split(torch.randperm(n), cfg.minibatch):
            dist, value = model(batch["states"][idx], batch["aux"][idx])
            logp = dist.log_prob(batch["actions"][idx]).sum(-1)
            ratio = (logp - batch["logp"][idx]).exp()
            adv = batch["adv"][idx]
            pg = -torch.min(
                ratio * adv,
                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv,
            ).mean()
            vf = (value - batch["returns"][idx]).pow(2).mean()
            ent = dist.entropy().sum(-1).mean()
            loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
            optim.zero_grad()
            loss.backward()
            optim.step()
    return float(loss.detach())


def load_data(data_path: str, fills: str, ticks_path: str) -> LOBDataset:
    df = load_lob(data_path)
    if fills == "tick":
        df = attach_tick_fills(df, ticks_path)
    return build_dataset(df, window=50, train_frac=TRAIN_FRAC)


def train(
    ds: LOBDataset, updates: int, seed: int, out_dir: Path
) -> tuple[ActorCritic, dict[str, list[float]]]:
    """Train on the first 70% of snapshots; log history and checkpoint."""
    torch.manual_seed(seed)
    n_train = int(len(ds.mid) * TRAIN_FRAC)
    env = MarketMakingEnv(
        ds, EnvConfig(), rng=np.random.default_rng(seed), start_range=(0, n_train)
    )
    model = ActorCritic()
    optim = torch.optim.Adam(model.parameters(), lr=PPOConfig.lr)
    cfg = PPOConfig()

    history: dict[str, list[float]] = {"ep_reward": [], "loss": [], "seconds": []}
    t0 = time.time()
    for u in range(updates):
        batch = collect_rollout(env, model, cfg)
        loss = ppo_update(model, optim, batch, cfg)
        history["ep_reward"].append(float(batch["ep_reward_mean"]))
        history["loss"].append(loss)
        history["seconds"].append(time.time() - t0)
        if (u + 1) % 25 == 0:
            r_recent = float(np.mean(history["ep_reward"][-25:]))
            print(
                f"update {u + 1:>4}/{updates} | mean ep_reward (25) "
                f"{r_recent:>10.2f} | {history['seconds'][-1]:.0f}s",
                flush=True,
            )
            torch.save(model.state_dict(), out_dir / "model.pt")
    torch.save(model.state_dict(), out_dir / "model.pt")
    return model, history


def model_policy(model: ActorCritic) -> Policy:
    """Deterministic policy: act with the Gaussian mean, no sampling."""

    def policy(env: MarketMakingEnv, res: StepResult) -> np.ndarray:
        with torch.no_grad():
            dist, _ = model(
                torch.from_numpy(res.state).unsqueeze(0),
                torch.from_numpy(res.aux).unsqueeze(0),
            )
        return np.asarray(dist.mean.numpy()[0])

    return policy


def fixed_policy(action1: float) -> Policy:
    """Symmetric quotes at a fixed spread level, no bias."""

    def policy(env: MarketMakingEnv, res: StepResult) -> np.ndarray:
        return np.array([0.0, action1], dtype=np.float32)

    return policy


def random_policy(rng: np.random.Generator) -> Policy:
    def policy(env: MarketMakingEnv, res: StepResult) -> np.ndarray:
        return rng.uniform(-1, 1, 2).astype(np.float32)

    return policy


def as_policy(ds: LOBDataset, gamma_skew_bps: float = 2.0) -> Policy:
    """Avellaneda-Stoikov closed form (paper eq. 17-18), calibrated.

    sigma^2 is estimated from 10s mid changes on the training split.
    gamma is set so that one unit of inventory skews the reservation
    price by ``gamma_skew_bps`` at episode start: gamma = skew/(sigma^2*T).
    kappa follows the A-S simulation convention (1.5).
    """
    n_train = int(len(ds.mid) * TRAIN_FRAC)
    sigma2 = float(np.var(np.diff(ds.mid[:n_train])))  # per-step price var
    mid_ref = float(np.median(ds.mid[:n_train]))
    kappa = 1.5

    def policy(env: MarketMakingEnv, res: StepResult) -> np.ndarray:
        cfg = env.cfg
        t_rem = cfg.episode_len - (env.t - env.start)
        gamma = gamma_skew_bps * 1e-4 * mid_ref / (sigma2 * cfg.episode_len)
        bias_price = -env.inventory * gamma * sigma2 * t_rem
        spread_price = gamma * sigma2 * t_rem + (2 / gamma) * np.log(1 + gamma / kappa)
        mid = env.data.mid[env.t]
        bias_bps = bias_price / mid * 1e4
        half_bps = spread_price / 2 / mid * 1e4
        a0 = np.clip(bias_bps / cfg.max_bias_bps, -1, 1)
        a1 = np.clip(
            2
            * (half_bps - cfg.min_half_spread_bps)
            / (cfg.max_half_spread_bps - cfg.min_half_spread_bps)
            - 1,
            -1,
            1,
        )
        return np.array([a0, a1], dtype=np.float32)

    return policy


def evaluate(ds: LOBDataset, policy: Policy, starts: list[int]) -> dict[str, float]:
    """Run fixed test episodes; report the paper's headline metrics."""
    env = MarketMakingEnv(ds, EnvConfig(), start_range=(0, len(ds.mid)))
    pnls, fills, abs_inv, spreads = [], [], [], []
    for s in starts:
        res = env.reset(start=s)
        ep_fills = 0.0
        inv_path = []
        while not res.done:
            res = env.step(policy(env, res))
            ep_fills += res.info["fills"]
            inv_path.append(abs(res.info["inventory"]))
        pnls.append(env.cash)  # inventory already liquidated at episode end
        fills.append(ep_fills)
        abs_inv.append(float(np.mean(inv_path)))
        spreads.append(float(np.mean(ds.spread[s : s + env.cfg.episode_len])))

    pnl = np.array(pnls)
    mean_abs_pos = float(np.mean(abs_inv))
    return {
        "episodes": len(starts),
        "pnl_mean": float(pnl.mean()),
        "pnl_std": float(pnl.std()),
        "pnl_total": float(pnl.sum()),
        "sharpe": float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0,
        "nd_pnl": float(pnl.mean() / np.mean(spreads)),
        "pnl_map": float(pnl.mean() / mean_abs_pos) if mean_abs_pos > 0 else 0.0,
        "fills_per_ep": float(np.mean(fills)),
        "mean_abs_inventory": mean_abs_pos,
    }


def test_starts(ds: LOBDataset, episode_len: int = 360) -> list[int]:
    """Non-overlapping episode starts covering the held-out last 30%."""
    n = len(ds.mid)
    lo = int(n * TRAIN_FRAC)
    return list(range(lo, n - episode_len - 1, episode_len))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/btc_usdt_20221019_20221030_lob.parquet")
    parser.add_argument(
        "--ticks", default="data/btc_usdt_20221019_20221030_ticks.parquet"
    )
    parser.add_argument("--fills", choices=["snapshot", "tick"], default="tick")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/run")
    parser.add_argument("--benchmarks-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ds = load_data(args.data, args.fills, args.ticks)
    print(f"data ready in {time.time() - t0:.0f}s | fills={args.fills}", flush=True)
    starts = test_starts(ds)

    results: dict[str, dict[str, float]] = {}
    meta: dict[str, object] = {
        "fills": args.fills,
        "updates": 0,
        "seed": args.seed,
        "machine": platform.platform(),
        "test_episodes": len(starts),
    }

    if not args.benchmarks_only:
        model, history = train(ds, args.updates, args.seed, out_dir)
        meta["updates"] = args.updates
        meta["train_seconds"] = history["seconds"][-1]
        (out_dir / "history.json").write_text(json.dumps(history))
        results["c_ppo"] = evaluate(ds, model_policy(model), starts)

    results["fixed_tight"] = evaluate(ds, fixed_policy(-1.0), starts)
    results["fixed_mid"] = evaluate(ds, fixed_policy(-0.5), starts)
    results["fixed_wide"] = evaluate(ds, fixed_policy(0.0), starts)
    results["random"] = evaluate(ds, random_policy(np.random.default_rng(1)), starts)
    results["as_strategy"] = evaluate(ds, as_policy(ds), starts)

    (out_dir / "metrics.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2)
    )
    print(json.dumps({"meta": meta, "results": results}, indent=2))


if __name__ == "__main__":
    main()
