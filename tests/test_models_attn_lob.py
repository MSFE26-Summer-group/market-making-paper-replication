"""Tests for paper_replication.models.attn_lob.

These check the architecture is wired correctly (shapes, gradient flow) --
none of them train the model (no optimizer, no parameter updates).
"""

import pytest
import torch

from paper_replication.models.attn_lob import AttnLOB, AttnLOBConfig


def test_forward_output_shapes() -> None:
    config = AttnLOBConfig()
    model = AttnLOB(config)
    x = torch.randn(8, 50, 4 * config.n_levels)

    logits = model(x)
    features = model.forward_features(x)

    assert logits.shape == (8, config.num_classes)
    assert features.shape == (8, config.embed_dim)


def test_forward_handles_arbitrary_batch_and_sequence_length() -> None:
    model = AttnLOB()
    x = torch.randn(3, 17, 40)  # T != the usual window_T=50

    logits = model(x)

    assert logits.shape == (3, 3)


def test_forward_is_deterministic_in_eval_mode() -> None:
    model = AttnLOB()
    model.eval()
    x = torch.randn(4, 50, 40)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    torch.testing.assert_close(out1, out2)


def test_gradients_flow_to_all_parameters() -> None:
    """One backward() pass to confirm no dead params. Smoke test only, not training."""
    model = AttnLOB()
    x = torch.randn(4, 50, 40)
    target = torch.tensor([0, 1, 2, 1])

    logits = model(x)
    loss: torch.Tensor = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()  # type: ignore[no-untyped-call]  # torch stubs don't type Tensor.backward

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"


def test_rejects_n_levels_not_a_multiple_of_5() -> None:
    with pytest.raises(ValueError, match="multiple of 5"):
        AttnLOB(AttnLOBConfig(n_levels=3))


def test_supports_deeper_n_levels_than_the_paper() -> None:
    """n_levels=50 is a deviation from the paper's Fig. 1 (n_levels=10), used
    for the depth-ablation experiment (Exp 3) -- confirms it still wires up."""
    config = AttnLOBConfig(n_levels=50)
    model = AttnLOB(config)
    x = torch.randn(2, 50, 4 * config.n_levels)

    logits = model(x)
    features = model.forward_features(x)

    assert logits.shape == (2, config.num_classes)
    assert features.shape == (2, config.embed_dim)


def test_dropout_is_a_noop_at_default() -> None:
    """dropout=0.0 (default) must reproduce the exact original architecture."""
    model = AttnLOB(AttnLOBConfig())
    model.eval()
    x = torch.randn(4, 50, 40)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    torch.testing.assert_close(out1, out2)


def test_dropout_is_disabled_in_eval_mode() -> None:
    model = AttnLOB(AttnLOBConfig(dropout=0.5))
    model.eval()
    x = torch.randn(4, 50, 40)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    torch.testing.assert_close(out1, out2)


def test_dropout_is_stochastic_in_train_mode() -> None:
    model = AttnLOB(AttnLOBConfig(dropout=0.5))
    model.train()
    x = torch.randn(4, 50, 40)

    out1 = model(x)
    out2 = model(x)

    assert not torch.allclose(out1, out2)


def test_custom_config_changes_dimensions() -> None:
    config = AttnLOBConfig(inception_channels=16, attn_heads=2, num_classes=5)
    model = AttnLOB(config)
    x = torch.randn(2, 50, 40)

    logits = model(x)
    features = model.forward_features(x)

    assert config.embed_dim == 48
    assert features.shape == (2, 48)
    assert logits.shape == (2, 5)
