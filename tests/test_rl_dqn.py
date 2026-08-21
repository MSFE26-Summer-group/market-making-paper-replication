"""Smoke test for paper_replication.rl.dqn -- runs end to end on a tiny setup.

Confirms the training loop wires up and produces finite losses; this does
not test convergence (real training is exercised in the RL training
notebook against the actual dataset).
"""

import numpy as np

from paper_replication.models.attn_lob import AttnLOBConfig
from paper_replication.rl.config import RLConfig
from paper_replication.rl.dqn import DQNConfig, train_dqn
from paper_replication.rl.env import MarketMakingEnv
from paper_replication.rl.market_data import MarketDataset
from paper_replication.rl.networks import DuelingQNetwork, RLNetworkConfig


def _dataset() -> MarketDataset:
    n = 13
    rng = np.random.default_rng(0)
    mid = 100.0 + np.cumsum(rng.normal(scale=0.05, size=n))
    return MarketDataset(
        lob_state=np.zeros((n, 2, 3)),
        dynamic_state=rng.normal(size=(n, 2)).astype(np.float64),
        dynamic_feature_names=["f1", "f2"],
        best_ask=mid + 0.5,
        best_bid=mid - 0.5,
        mid_price=mid,
        timestamps=np.arange(n, dtype=float) * 10.0,
    )


def test_train_dqn_runs_and_produces_finite_losses() -> None:
    dataset = _dataset()
    config = RLConfig(
        minimum_trade_unit=0.01,
        omega=5.0,
        max_bias=0.05,
        max_spread=0.1,
        episode_length=3,
    )
    env = MarketMakingEnv(dataset, config, action_space_kind="discrete")

    net_config = RLNetworkConfig(
        attn_lob_config=AttnLOBConfig(n_levels=5, inception_channels=4, attn_heads=2),
        n_dynamic_features=2,
        n_agent_features=2,
        trunk_hidden_dim=8,
    )
    model = DuelingQNetwork(net_config, n_actions=8)
    lob_feature_cache = (
        np.random.default_rng(1)
        .normal(size=(len(dataset), net_config.attn_lob_config.embed_dim))
        .astype(np.float32)
    )

    dqn_config = DQNConfig(
        n_steps=40,
        warmup_steps=8,
        batch_size=4,
        buffer_size=50,
        train_interval=2,
        target_update_interval=10,
        log_interval=10,
        device="cpu",
    )

    history = train_dqn(env, model, lob_feature_cache, dqn_config)

    assert len(history) == 4  # n_steps // log_interval
    for entry in history:
        assert 0.0 <= entry["epsilon"] <= 1.0
        assert np.isfinite(entry["mean_recent_episode_reward"])


def test_train_dqn_restricts_rollouts_to_given_episode_indices() -> None:
    dataset = _dataset()
    config = RLConfig(
        minimum_trade_unit=0.01,
        omega=5.0,
        max_bias=0.05,
        max_spread=0.1,
        episode_length=3,
    )
    env = MarketMakingEnv(dataset, config, action_space_kind="discrete")

    net_config = RLNetworkConfig(
        attn_lob_config=AttnLOBConfig(n_levels=5, inception_channels=4, attn_heads=2),
        n_dynamic_features=2,
        n_agent_features=2,
        trunk_hidden_dim=8,
    )
    model = DuelingQNetwork(net_config, n_actions=8)
    lob_feature_cache = np.zeros(
        (len(dataset), net_config.attn_lob_config.embed_dim), dtype=np.float32
    )
    dqn_config = DQNConfig(
        n_steps=20,
        warmup_steps=5,
        batch_size=4,
        buffer_size=50,
        log_interval=10,
        device="cpu",
    )

    train_dqn(env, model, lob_feature_cache, dqn_config, train_episode_indices=[0])
