"""Tests for paper_replication.rl.feature_cache."""

import numpy as np
import torch

from paper_replication.models.attn_lob import AttnLOB, AttnLOBConfig
from paper_replication.rl.feature_cache import precompute_lob_features


def test_precompute_matches_direct_forward_features() -> None:
    config = AttnLOBConfig(n_levels=5, inception_channels=8, attn_heads=2)
    model = AttnLOB(config)
    model.eval()
    lob_state = np.random.default_rng(0).normal(size=(10, 6, 4 * config.n_levels))

    cached = precompute_lob_features(model, lob_state, device="cpu", batch_size=3)

    with torch.no_grad():
        expected = model.forward_features(torch.from_numpy(lob_state).float())

    np.testing.assert_allclose(cached, expected.numpy(), atol=1e-5, rtol=1e-4)


def test_precompute_output_shape_and_dtype() -> None:
    config = AttnLOBConfig(n_levels=5)
    model = AttnLOB(config)
    lob_state = np.zeros((4, 6, 4 * config.n_levels))

    cached = precompute_lob_features(model, lob_state, device="cpu")

    assert cached.shape == (4, config.embed_dim)
    assert cached.dtype == np.float32


def test_precompute_restores_model_training_mode() -> None:
    config = AttnLOBConfig(n_levels=5)
    model = AttnLOB(config)
    model.train()
    lob_state = np.zeros((2, 6, 4 * config.n_levels))

    precompute_lob_features(model, lob_state, device="cpu")

    assert model.training is True
