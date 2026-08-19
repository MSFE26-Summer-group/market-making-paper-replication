"""Tests for paper_replication.rl.networks.

Shape/gradient-flow smoke tests only, matching test_models_attn_lob.py's
style -- no optimizer, no parameter updates.
"""

import torch

from paper_replication.models.attn_lob import AttnLOBConfig
from paper_replication.rl.networks import (
    ContinuousActorCritic,
    DuelingQNetwork,
    RLFeatureExtractor,
    RLNetworkConfig,
)


def _small_config() -> RLNetworkConfig:
    return RLNetworkConfig(
        attn_lob_config=AttnLOBConfig(n_levels=5, inception_channels=8, attn_heads=2),
        n_dynamic_features=6,
        n_agent_features=2,
        trunk_hidden_dim=16,
    )


def _inputs(
    config: RLNetworkConfig, batch: int = 4, window_t: int = 10
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lob_state = torch.randn(batch, window_t, 4 * config.attn_lob_config.n_levels)
    dynamic_state = torch.randn(batch, config.n_dynamic_features)
    agent_state = torch.randn(batch, config.n_agent_features)
    return lob_state, dynamic_state, agent_state


def test_feature_extractor_from_lob_features_matches_full_forward() -> None:
    """Precomputed-feature path (used by ppo.py/dqn.py) must equal the full path."""
    config = _small_config()
    extractor = RLFeatureExtractor(config)
    extractor.eval()
    lob_state, dynamic_state, agent_state = _inputs(config)

    with torch.no_grad():
        full = extractor(lob_state, dynamic_state, agent_state)
        lob_features = extractor.attn_lob.forward_features(lob_state)
        cached = extractor.forward_from_lob_features(
            lob_features, dynamic_state, agent_state
        )

    torch.testing.assert_close(full, cached)


def test_continuous_actor_critic_from_lob_features_matches_full_forward() -> None:
    config = _small_config()
    model = ContinuousActorCritic(config)
    model.eval()
    lob_state, dynamic_state, agent_state = _inputs(config)

    with torch.no_grad():
        alpha_full, beta_full, value_full = model(lob_state, dynamic_state, agent_state)
        lob_features = model.extractor.attn_lob.forward_features(lob_state)
        alpha_cached, beta_cached, value_cached = model.forward_from_lob_features(
            lob_features, dynamic_state, agent_state
        )

    torch.testing.assert_close(alpha_full, alpha_cached)
    torch.testing.assert_close(beta_full, beta_cached)
    torch.testing.assert_close(value_full, value_cached)


def test_dueling_q_network_from_lob_features_matches_full_forward() -> None:
    config = _small_config()
    model = DuelingQNetwork(config)
    model.eval()
    lob_state, dynamic_state, agent_state = _inputs(config)

    with torch.no_grad():
        q_full = model(lob_state, dynamic_state, agent_state)
        lob_features = model.extractor.attn_lob.forward_features(lob_state)
        q_cached = model.forward_from_lob_features(
            lob_features, dynamic_state, agent_state
        )

    torch.testing.assert_close(q_full, q_cached)


def test_feature_extractor_output_shape() -> None:
    config = _small_config()
    extractor = RLFeatureExtractor(config)
    lob_state, dynamic_state, agent_state = _inputs(config)

    out = extractor(lob_state, dynamic_state, agent_state)

    assert out.shape == (4, config.trunk_hidden_dim)


def test_continuous_actor_critic_output_shapes() -> None:
    config = _small_config()
    model = ContinuousActorCritic(config)
    lob_state, dynamic_state, agent_state = _inputs(config)

    alpha, beta, value = model(lob_state, dynamic_state, agent_state)

    assert alpha.shape == (4, 2)
    assert beta.shape == (4, 2)
    assert value.shape == (4,)


def test_continuous_actor_critic_beta_params_are_greater_than_one() -> None:
    """alpha, beta > 1 keeps the Beta distribution unimodal (see module docstring)."""
    config = _small_config()
    model = ContinuousActorCritic(config)
    lob_state, dynamic_state, agent_state = _inputs(config)

    alpha, beta, _ = model(lob_state, dynamic_state, agent_state)

    assert torch.all(alpha > 1.0)
    assert torch.all(beta > 1.0)


def test_dueling_q_network_output_shape() -> None:
    config = _small_config()
    model = DuelingQNetwork(config, n_actions=8)
    lob_state, dynamic_state, agent_state = _inputs(config)

    q_values = model(lob_state, dynamic_state, agent_state)

    assert q_values.shape == (4, 8)


def test_gradients_flow_through_continuous_actor_critic() -> None:
    config = _small_config()
    model = ContinuousActorCritic(config)
    lob_state, dynamic_state, agent_state = _inputs(config)

    alpha, beta, value = model(lob_state, dynamic_state, agent_state)
    loss = alpha.sum() + beta.sum() + value.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if "attn_lob.pretrain_head" in name:
            continue  # unused here: RL heads only call forward_features
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"


def test_gradients_flow_through_dueling_q_network() -> None:
    config = _small_config()
    model = DuelingQNetwork(config)
    lob_state, dynamic_state, agent_state = _inputs(config)

    q_values = model(lob_state, dynamic_state, agent_state)
    q_values.sum().backward()

    for name, param in model.named_parameters():
        if "attn_lob.pretrain_head" in name:
            continue  # unused here: RL heads only call forward_features
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"


def test_dueling_q_network_advantage_is_zero_mean_centered() -> None:
    """Sanity check on the dueling decomposition: Q - V should average to ~0 per row."""
    config = _small_config()
    model = DuelingQNetwork(config)
    model.eval()
    lob_state, dynamic_state, agent_state = _inputs(config)

    with torch.no_grad():
        features = model.extractor(lob_state, dynamic_state, agent_state)
        value = model.value_head(features)
        q_values = model(lob_state, dynamic_state, agent_state)

    torch.testing.assert_close(
        (q_values - value).mean(dim=-1), torch.zeros(4), atol=1e-5, rtol=1e-4
    )
