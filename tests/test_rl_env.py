"""Tests for paper_replication.rl.env.

Uses a hand-built `MarketDataset` with `max_bias=max_spread=0` so every
decoded quote collapses to `bid=ask=mid_price` regardless of the action
value -- this isolates the environment's stepping/fill/episode wiring from
`action_space.py`'s own decode logic (already covered by
`test_rl_action_space.py`).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from paper_replication.rl.config import RLConfig
from paper_replication.rl.env import MarketMakingEnv
from paper_replication.rl.market_data import MarketDataset
from paper_replication.rl.simulator import Fill


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


def test_reset_returns_flat_agent_state_at_episode_start() -> None:
    env = MarketMakingEnv(_dataset(), _config(), action_space_kind="continuous")

    obs = env.reset(episode_index=0)

    np.testing.assert_allclose(obs.agent_state, [0.0, 0.0])
    np.testing.assert_allclose(obs.lob_state, env.dataset.lob_state[0])


def test_current_row_tracks_the_returned_observation() -> None:
    env = MarketMakingEnv(_dataset(), _config(), action_space_kind="continuous")

    env.reset(episode_index=0)
    assert env.current_row == 0

    result = env.step(np.array([0.5, 0.5]))
    assert env.current_row == 1
    assert result.observation is not None
    np.testing.assert_allclose(result.observation.lob_state, env.dataset.lob_state[1])


def test_continuous_episode_fills_and_closes_as_expected() -> None:
    env = MarketMakingEnv(_dataset(), _config(), action_space_kind="continuous")
    env.reset(episode_index=0)
    action = np.array([0.5, 0.5])  # irrelevant: max_bias=max_spread=0

    # Step 0: bid=ask=mid_price[0]=100; next row's best_ask=99 -> buy fill @100.
    result0 = env.step(action)
    assert result0.done is False
    assert result0.info["inventory"] == pytest.approx(1.0)
    assert result0.info["cash"] == pytest.approx(-100.0)
    assert len(cast("list[Fill]", result0.info["fills"])) == 1

    # Step 1: bid=ask=mid_price[1]=99; next row's best_bid=100 -> sell fill @99.
    result1 = env.step(action)
    assert result1.done is False
    assert result1.info["inventory"] == pytest.approx(0.0)
    assert result1.info["cash"] == pytest.approx(-1.0)

    # Step 2 (last step of episode_length=3): forced close, already flat -> no-op.
    result2 = env.step(action)
    assert result2.done is True
    assert result2.observation is None
    assert result2.info["inventory"] == pytest.approx(0.0)
    assert result2.info["cash"] == pytest.approx(-1.0)
    assert result2.info["fills"] == []


def test_discrete_action_space_wires_up_the_same_way() -> None:
    env = MarketMakingEnv(_dataset(), _config(), action_space_kind="discrete")
    env.reset(episode_index=0)

    result0 = env.step(0)  # levels[0] fractions are moot: max_bias=max_spread=0
    assert result0.info["inventory"] == pytest.approx(1.0)

    result1 = env.step(0)
    assert result1.info["inventory"] == pytest.approx(0.0)


def test_continuous_env_rejects_int_action() -> None:
    env = MarketMakingEnv(_dataset(), _config(), action_space_kind="continuous")
    env.reset(episode_index=0)

    with pytest.raises(AssertionError):
        env.step(0)


def test_discrete_env_rejects_array_action() -> None:
    env = MarketMakingEnv(_dataset(), _config(), action_space_kind="discrete")
    env.reset(episode_index=0)

    with pytest.raises(AssertionError):
        env.step(np.array([0.5, 0.5]))


def test_reset_out_of_range_episode_index_raises() -> None:
    env = MarketMakingEnv(_dataset(), _config(), action_space_kind="continuous")

    with pytest.raises(ValueError, match="episode_index"):
        env.reset(episode_index=5)


def test_construction_rejects_dataset_too_short_for_one_episode() -> None:
    dataset = _dataset()  # 4 rows
    config = RLConfig(episode_length=10)  # needs 11 rows

    with pytest.raises(ValueError, match="dataset has"):
        MarketMakingEnv(dataset, config)


def test_reset_without_episode_index_advances_sequentially() -> None:
    n = 7  # two full episodes of length 3 fit (needs 3*2 + 1 = 7 rows)
    dataset = MarketDataset(
        lob_state=np.zeros((n, 1, 1)),
        dynamic_state=np.zeros((n, 1)),
        dynamic_feature_names=["f1"],
        best_ask=np.full(n, 101.0),
        best_bid=np.full(n, 99.0),
        mid_price=np.arange(n, dtype=float) + 100.0,
        timestamps=np.arange(n, dtype=float),
    )
    env = MarketMakingEnv(dataset, _config(), action_space_kind="continuous")

    first = env.reset()
    second = env.reset()

    assert env.n_episodes == 2
    np.testing.assert_allclose(first.lob_state, dataset.lob_state[0])
    np.testing.assert_allclose(second.lob_state, dataset.lob_state[3])
