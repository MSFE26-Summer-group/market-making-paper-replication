"""Smoke test for paper_replication.rl.ppo -- runs end to end on a tiny setup.

Confirms the training loop wires up and produces finite losses; this does
not test convergence (real training is exercised in the RL training
notebook against the actual dataset).
"""

import numpy as np

from paper_replication.models.attn_lob import AttnLOBConfig
from paper_replication.rl.config import RLConfig
from paper_replication.rl.env import MarketMakingEnv
from paper_replication.rl.market_data import MarketDataset
from paper_replication.rl.networks import ContinuousActorCritic, RLNetworkConfig
from paper_replication.rl.ppo import PPOConfig, train_ppo


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


def test_train_ppo_runs_and_produces_finite_losses() -> None:
    dataset = _dataset()
    config = RLConfig(
        minimum_trade_unit=0.01,
        omega=5.0,
        max_bias=0.05,
        max_spread=0.1,
        episode_length=3,
    )
    env = MarketMakingEnv(dataset, config, action_space_kind="continuous")

    net_config = RLNetworkConfig(
        attn_lob_config=AttnLOBConfig(n_levels=5, inception_channels=4, attn_heads=2),
        n_dynamic_features=2,
        n_agent_features=2,
        trunk_hidden_dim=8,
    )
    model = ContinuousActorCritic(net_config)
    lob_feature_cache = (
        np.random.default_rng(1)
        .normal(size=(len(dataset), net_config.attn_lob_config.embed_dim))
        .astype(np.float32)
    )

    ppo_config = PPOConfig(
        n_updates=2,
        episodes_per_update=2,
        update_epochs=2,
        minibatch_size=4,
        device="cpu",
    )

    history = train_ppo(env, model, lob_feature_cache, ppo_config)

    assert len(history) == 2
    for entry in history:
        assert np.isfinite(entry["policy_loss"])
        assert np.isfinite(entry["value_loss"])
        assert np.isfinite(entry["mean_episode_reward"])


def test_train_ppo_restricts_rollouts_to_given_episode_indices() -> None:
    dataset = _dataset()
    config = RLConfig(
        minimum_trade_unit=0.01,
        omega=5.0,
        max_bias=0.05,
        max_spread=0.1,
        episode_length=3,
    )
    env = MarketMakingEnv(dataset, config, action_space_kind="continuous")
    assert env.n_episodes == 4

    net_config = RLNetworkConfig(
        attn_lob_config=AttnLOBConfig(n_levels=5, inception_channels=4, attn_heads=2),
        n_dynamic_features=2,
        n_agent_features=2,
        trunk_hidden_dim=8,
    )
    model = ContinuousActorCritic(net_config)
    lob_feature_cache = np.zeros(
        (len(dataset), net_config.attn_lob_config.embed_dim), dtype=np.float32
    )

    ppo_config = PPOConfig(
        n_updates=1, episodes_per_update=2, minibatch_size=4, device="cpu"
    )

    # Should not raise even when restricted to a single episode index.
    train_ppo(env, model, lob_feature_cache, ppo_config, train_episode_indices=[0])
