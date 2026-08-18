"""Model-agnostic training loop for the pretrain classification task (paper IV-B).

Works with any of `paper_replication.models`'s pretrain models (AttnLOB,
FCLOB, ConvLOB, DeepLOB) -- they all share the same `forward(lob_state) ->
logits` signature, so nothing here needs to know which one it's training.
`num_classes` is passed explicitly rather than read off a model-specific
`.config` attribute, keeping this file free of any per-model coupling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from paper_replication.features.dataset import AttnLOBDataset
from paper_replication.training.checkpoint import save_checkpoint
from paper_replication.training.config import TrainConfig
from paper_replication.training.dataset import AttnLOBTorchDataset
from paper_replication.training.metrics import confusion_matrix, precision_recall_f1


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Runs one training epoch; returns the sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits: torch.Tensor = model(x)
        loss: torch.Tensor = F.cross_entropy(logits, y)
        loss.backward()  # type: ignore[no-untyped-call]  # torch stubs don't type Tensor.backward
        optimizer.step()

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        n_samples += batch_size
    return total_loss / n_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    """Sample-weighted mean loss plus macro precision/recall/F1/accuracy."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_true = []
    all_pred = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits: torch.Tensor = model(x)
        loss: torch.Tensor = F.cross_entropy(logits, y)

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        n_samples += batch_size
        all_true.append(y)
        all_pred.append(logits.argmax(dim=-1))

    confusion = confusion_matrix(torch.cat(all_true), torch.cat(all_pred), num_classes)
    metrics = precision_recall_f1(confusion)
    metrics["val_loss"] = total_loss / n_samples
    return metrics


def train_model(
    model: nn.Module,
    train_dataset: AttnLOBDataset,
    val_dataset: AttnLOBDataset,
    config: TrainConfig,
    num_classes: int,
) -> list[dict[str, float]]:
    """Trains `model` in place; returns one metrics dict per epoch.

    Stops after `config.patience` consecutive epochs with no val_loss
    improvement (disabled if `patience` is None, in which case it always
    runs all `config.epochs`). Keeps the weights from whichever epoch had
    the lowest validation loss -- not necessarily the last epoch run -- and
    loads them back into `model` before returning. If `config.checkpoint_path`
    is set, those best weights are also saved there via `checkpoint.save_checkpoint`.
    """
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        AttnLOBTorchDataset(train_dataset),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        AttnLOBTorchDataset(val_dataset),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, num_classes)
        history.append({"epoch": float(epoch), "train_loss": train_loss, **val_metrics})

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                config.patience is not None
                and epochs_without_improvement >= config.patience
            ):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if config.checkpoint_path is not None:
        save_checkpoint(model, config.checkpoint_path)

    return history
