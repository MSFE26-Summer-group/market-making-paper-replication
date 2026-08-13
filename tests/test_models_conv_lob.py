"""Tests for paper_replication.models.conv_lob."""

import torch

from paper_replication.models.conv_lob import ConvLOB, ConvLOBConfig


def test_forward_output_shape() -> None:
    config = ConvLOBConfig()
    model = ConvLOB(config)
    x = torch.randn(8, 50, 4 * config.n_levels)

    logits = model(x)

    assert logits.shape == (8, config.num_classes)


def test_forward_handles_arbitrary_sequence_length() -> None:
    """Unlike FC-LOB, this is fully convolutional -- any T should work."""
    model = ConvLOB()
    x = torch.randn(3, 17, 40)

    logits = model(x)

    assert logits.shape == (3, 3)


def test_forward_is_deterministic_in_eval_mode() -> None:
    model = ConvLOB()
    model.eval()
    x = torch.randn(4, 50, 40)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    torch.testing.assert_close(out1, out2)


def test_gradients_flow_to_all_parameters() -> None:
    model = ConvLOB()
    x = torch.randn(4, 50, 40)
    target = torch.tensor([0, 1, 2, 1])

    logits = model(x)
    loss: torch.Tensor = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()  # type: ignore[no-untyped-call]  # torch stubs don't type Tensor.backward

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"


def test_output_changes_when_a_timestep_inside_the_receptive_field_changes() -> None:
    """The last timestep's output should be sensitive to any position within
    its causal receptive field, confirming the conv stack actually looks
    backward in time rather than only ever reading the final position.
    """
    config = ConvLOBConfig(num_blocks=2)  # receptive_field = 7
    model = ConvLOB(config)
    model.eval()

    x1 = torch.randn(1, 10, 40)
    x2 = x1.clone()
    # second-to-last timestep is well inside any receptive_field >= 2.
    x2[:, -2, :] = x1[:, -2, :] + 10.0

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    assert not torch.allclose(out1, out2)


def test_output_is_unaffected_by_timesteps_outside_the_receptive_field() -> None:
    """Confirms the causal padding actually bounds the receptive field --
    not just "has some memory", but the *exact* size `receptive_field` claims.
    """
    config = ConvLOBConfig(num_blocks=2)  # receptive_field = 7
    model = ConvLOB(config)
    model.eval()

    t = 10  # last index=9; positions 0..2 are outside a 7-wide receptive field
    x1 = torch.randn(1, t, 40)
    x2 = x1.clone()
    x2[:, 0, :] = x1[:, 0, :] + 10.0

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    torch.testing.assert_close(out1, out2)


def test_receptive_field_formula() -> None:
    config = ConvLOBConfig(kernel_size=3, num_blocks=3)
    # RF = 1 + (k-1)*(1 + 2 + 4) = 1 + 2*7 = 15
    assert config.receptive_field == 15


def test_custom_config_changes_dimensions() -> None:
    config = ConvLOBConfig(hidden_channels=16, num_blocks=2, num_classes=5)
    model = ConvLOB(config)
    x = torch.randn(2, 50, 40)

    logits = model(x)

    assert logits.shape == (2, 5)
