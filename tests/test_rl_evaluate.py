"""Tests for paper_replication.rl.evaluate."""

import numpy as np

from paper_replication.rl.config import RLConfig
from paper_replication.rl.env import MarketMakingEnv
from paper_replication.rl.evaluate import evaluate_policy, run_episode
from paper_replication.rl.market_data import MarketDataset


def _dataset() -> MarketDataset:
    n = 4
    return MarketDataset(
        lob_state=np.zeros((n, 2, 3)),
        dynamic_state=np.zeros((n, 2)),
        dynamic_feature_names=["f1", "f2"],
        best_ask=np.array([102.0, 99.0, 102.0, 104.0]),
        best_bid=np.array([98.0, 97.0, 100.0, 103.0]),
        mid_price=np.array([100.0, 99.0, 101.0, 102.0]),
        timestamps=np.arange(n, dtype=float) * 10.0,
    )


def _config() -> RLConfig:
    return RLConfig(
        minimum_trade_unit=1.0,
        omega=5.0,
        initial_cash=0.0,
        max_bias=0.0,
        max_spread=0.0,
        eta=0.0,
        zeta=0.0,
        episode_length=3,
    )


def _zero_policy(observation: object, row: int) -> np.ndarray:
    return np.array([0.5, 0.5])


def test_run_episode_records_final_pnl_and_fills() -> None:
    env = MarketMakingEnv(_dataset(), _config(), action_space_kind="continuous")

    log = run_episode(env, episode_index=0, policy=_zero_policy)

    # Matches test_rl_env.py's hand-derived trajectory: buy@100, sell@99 -> cash=-1.
    assert log.final_pnl == -1.0
    assert len(log.fills) == 2
    assert log.abs_positions[0] == 0.0  # starts flat


def test_evaluate_policy_aggregates_across_episode_indices() -> None:
    env = MarketMakingEnv(_dataset(), _config(), action_space_kind="continuous")

    metrics = evaluate_policy(env, _zero_policy, episode_indices=[0, 0])

    assert metrics.n_episodes == 2
