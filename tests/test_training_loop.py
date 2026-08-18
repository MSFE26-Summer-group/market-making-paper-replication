"""Tests for paper_replication.training.loop and .checkpoint.

`_trivial_dataset` gives each class a widely separated constant offset --
a pattern a correctly-wired model+loop should fit almost perfectly in a
handful of epochs. These tests check the *loop* is correct (gradients
flow, the optimizer actually updates weights, loss goes down, checkpoints
round-trip) using a tiny model/dataset for speed; they are not a claim
about learning anything from real market data.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from paper_replication.features.dataset import AttnLOBDataset
from paper_replication.models.attn_lob import AttnLOB, AttnLOBConfig
from paper_replication.training.checkpoint import load_checkpoint, save_checkpoint
from paper_replication.training.config import TrainConfig
from paper_replication.training.loop import train_model


def _trivial_dataset(n_per_class: int = 20, seed: int = 0) -> AttnLOBDataset:
    rng = np.random.default_rng(seed)
    labels = np.array(
        [-1] * n_per_class + [0] * n_per_class + [1] * n_per_class, dtype=np.int8
    )
    offsets = {-1: -5.0, 0: 0.0, 1: 5.0}
    lob_state = np.stack(
        [
            rng.normal(loc=offsets[int(label)], scale=0.1, size=(50, 40))
            for label in labels
        ]
    )
    return AttnLOBDataset(
        lob_state=lob_state,
        dynamic_state=np.zeros((len(labels), 1)),
        dynamic_feature_names=["dummy"],
        labels=labels,
        timestamps=np.arange(len(labels), dtype=np.float64),
        alpha_used=0.0,
    )


def _tiny_model() -> AttnLOB:
    return AttnLOB(AttnLOBConfig(conv_hidden_dim=8, inception_channels=8, attn_heads=2))


def test_train_model_fits_trivial_pattern() -> None:
    dataset = _trivial_dataset()
    # patience=None: this test wants a fixed 15-epoch run to check the loss
    # curve; early stopping (covered separately below) could otherwise cut
    # it short once loss flattens out near zero on this easy a pattern.
    # device="cpu": keep this deterministic/portable regardless of what
    # machine runs it (TrainConfig defaults to CUDA when one is visible).
    config = TrainConfig(
        batch_size=16,
        epochs=15,
        learning_rate=5e-3,
        seed=0,
        patience=None,
        device="cpu",
    )
    model = _tiny_model()

    history = train_model(model, dataset, dataset, config, num_classes=3)

    assert len(history) == config.epochs
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert history[-1]["accuracy"] > 0.9


def test_train_model_keeps_lowest_val_loss_weights() -> None:
    dataset = _trivial_dataset()
    config = TrainConfig(
        batch_size=16, epochs=10, learning_rate=5e-3, seed=0, device="cpu"
    )
    model = _tiny_model()

    history = train_model(model, dataset, dataset, config, num_classes=3)

    best_val_loss = min(record["val_loss"] for record in history)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(dataset.lob_state).float()
        y = torch.from_numpy(dataset.labels.astype(np.int64) + 1)
        final_loss = torch.nn.functional.cross_entropy(model(x), y).item()

    assert final_loss == best_val_loss or final_loss <= best_val_loss + 1e-4


def test_checkpoint_round_trip_reproduces_outputs(tmp_path: Path) -> None:
    model = _tiny_model()
    model.eval()
    checkpoint_path = tmp_path / "attn_lob.pt"

    save_checkpoint(model, str(checkpoint_path))
    loaded = load_checkpoint(str(checkpoint_path))
    loaded.eval()

    x = torch.randn(4, 50, 40)
    with torch.no_grad():
        original_out = model(x)
        loaded_out = loaded(x)

    torch.testing.assert_close(original_out, loaded_out)


def test_save_checkpoint_rejects_unregistered_model_class(tmp_path: Path) -> None:
    class _UnregisteredModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = object()

    with pytest.raises(ValueError, match="_UnregisteredModel"):
        save_checkpoint(_UnregisteredModel(), str(tmp_path / "bad.pt"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_load_checkpoint_honors_map_location_cuda(tmp_path: Path) -> None:
    """Regression test: load_state_dict copies in place onto the fresh
    model's own (CPU) tensors -- map_location alone doesn't move the
    returned model, only the intermediate loaded tensors. Only meaningful
    (and only runs) on a machine with a visible GPU.
    """
    model = _tiny_model()
    checkpoint_path = tmp_path / "attn_lob.pt"
    save_checkpoint(model, str(checkpoint_path))

    loaded = load_checkpoint(str(checkpoint_path), map_location="cuda")

    assert all(p.is_cuda for p in loaded.parameters())


def test_train_model_writes_checkpoint_file(tmp_path: Path) -> None:
    dataset = _trivial_dataset(n_per_class=5)
    checkpoint_path = tmp_path / "attn_lob.pt"
    config = TrainConfig(
        batch_size=8,
        epochs=2,
        patience=None,
        device="cpu",
        checkpoint_path=str(checkpoint_path),
    )
    model = _tiny_model()

    train_model(model, dataset, dataset, config, num_classes=3)

    assert checkpoint_path.exists()
    loaded = load_checkpoint(str(checkpoint_path))
    assert loaded.config == model.config


def _patch_evaluate_with_fixed_sequence(
    monkeypatch: pytest.MonkeyPatch, val_losses: list[float]
) -> None:
    """Replaces loop.evaluate with one that returns a scripted val_loss sequence.

    Decouples early-stopping tests from actual model learning dynamics --
    they're testing the stop/continue bookkeeping, not whether the tiny
    model happens to plateau on cue.
    """
    losses = iter(val_losses)

    def fake_evaluate(
        model: nn.Module, loader: Any, device: torch.device, num_classes: int
    ) -> dict[str, float]:
        return {
            "val_loss": next(losses),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }

    monkeypatch.setattr("paper_replication.training.loop.evaluate", fake_evaluate)


def test_train_model_stops_early_when_val_loss_plateaus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _trivial_dataset(n_per_class=5)
    config = TrainConfig(batch_size=8, epochs=20, patience=3, device="cpu")
    model = _tiny_model()

    # improves through epoch 2 (best=0.5), then plateaus -> 3 non-improving
    # epochs (3, 4, 5) trip patience=3, so training stops after epoch 5.
    _patch_evaluate_with_fixed_sequence(
        monkeypatch, [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    )

    history = train_model(model, dataset, dataset, config, num_classes=3)

    assert len(history) == 5
    assert history[-1]["epoch"] == 5


def test_train_model_patience_none_disables_early_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _trivial_dataset(n_per_class=5)
    config = TrainConfig(batch_size=8, epochs=4, patience=None, device="cpu")
    model = _tiny_model()

    _patch_evaluate_with_fixed_sequence(monkeypatch, [1.0, 1.0, 1.0, 1.0])

    history = train_model(model, dataset, dataset, config, num_classes=3)

    assert len(history) == config.epochs
