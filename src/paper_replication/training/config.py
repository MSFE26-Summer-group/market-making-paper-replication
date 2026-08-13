"""Hyperparameters for the Attn-LOB pretraining loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters.

    epochs: an upper bound, not a target -- with `patience` set, training
        usually stops well before reaching it.
    patience: stop after this many consecutive epochs with no val_loss
        improvement. `None` disables early stopping (always run all
        `epochs`). Default 5 is deliberately tight: in practice this model
        has overfit (val_loss rising while train_loss keeps falling) within
        a handful of epochs on the dataset this was built against.
    device: defaults to "cuda" if a GPU is visible at import time, else
        "cpu" -- pass an explicit value to override either way.
    checkpoint_path: if set, the best-validation-loss model weights are
        saved here (via `paper_replication.training.checkpoint.save_checkpoint`)
        once training finishes.
    """

    batch_size: int = 256
    epochs: int = 50
    patience: int | None = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = field(default_factory=_default_device)
    num_workers: int = 0
    seed: int = 0
    checkpoint_path: str | None = None
