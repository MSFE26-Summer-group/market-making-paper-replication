"""Tests for paper_replication.models.fc_lob."""

import pytest
import torch

from paper_replication.models.fc_lob import FCLOB, FCLOBConfig


def test_forward_output_shape() -> None:
    config = FCLOBConfig()
    model = FCLOB(config)
    x = torch.randn(8, config.window_T, 4 * config.n_levels)

    logits = model(x)

    assert logits.shape == (8, config.num_classes)


def test_forward_rejects_mismatched_window_t() -> None:
    model = FCLOB(FCLOBConfig(window_T=50))
    x = torch.randn(4, 17, 40)  # wrong T

    with pytest.raises(ValueError):
        model(x)


def test_forward_is_deterministic_in_eval_mode() -> None:
    model = FCLOB()
    model.eval()
    x = torch.randn(4, 50, 40)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    torch.testing.assert_close(out1, out2)


def test_gradients_flow_to_all_parameters() -> None:
    model = FCLOB()
    x = torch.randn(4, 50, 40)
    target = torch.tensor([0, 1, 2, 1])

    logits = model(x)
    loss: torch.Tensor = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()  # type: ignore[no-untyped-call]  # torch stubs don't type Tensor.backward

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"


def test_custom_config_changes_dimensions() -> None:
    config = FCLOBConfig(n_levels=10, window_T=20, hidden_dims=(64, 32), num_classes=5)
    model = FCLOB(config)
    x = torch.randn(2, 20, 40)

    logits = model(x)

    assert config.input_dim == 800
    assert logits.shape == (2, 5)
