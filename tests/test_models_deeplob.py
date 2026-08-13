"""Tests for paper_replication.models.deeplob."""

import pytest
import torch

from paper_replication.models.deeplob import DeepLOB, DeepLOBConfig


def test_forward_output_shapes() -> None:
    config = DeepLOBConfig()
    model = DeepLOB(config)
    x = torch.randn(8, 50, 4 * config.n_levels)

    logits = model(x)
    features = model.forward_features(x)

    assert logits.shape == (8, config.num_classes)
    assert features.shape == (8, config.lstm_hidden_dim)


def test_forward_handles_arbitrary_batch_and_sequence_length() -> None:
    model = DeepLOB()
    x = torch.randn(3, 17, 40)  # T != the usual window_T=50

    logits = model(x)

    assert logits.shape == (3, 3)


def test_forward_is_deterministic_in_eval_mode() -> None:
    model = DeepLOB()
    model.eval()
    x = torch.randn(4, 50, 40)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    torch.testing.assert_close(out1, out2)


def test_gradients_flow_to_all_parameters() -> None:
    model = DeepLOB()
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
        DeepLOB(DeepLOBConfig(n_levels=3))


def test_custom_config_changes_dimensions() -> None:
    config = DeepLOBConfig(inception_channels=16, lstm_hidden_dim=8, num_classes=5)
    model = DeepLOB(config)
    x = torch.randn(2, 50, 40)

    logits = model(x)
    features = model.forward_features(x)

    assert config.lstm_input_dim == 48
    assert features.shape == (2, 8)
    assert logits.shape == (2, 5)
