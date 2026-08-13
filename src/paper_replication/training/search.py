"""Small hyperparameter search harness for pretrain-model training.

Trains each candidate on `train_dataset`/`val_dataset` only and ranks
candidates by their best (lowest) val_loss -- the test split never appears
here. The intended usage is: run the search, inspect `SearchResult.history`
for the winner, then evaluate that one model on the held-out test split
exactly once, outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from torch import nn

from paper_replication.features.dataset import AttnLOBDataset
from paper_replication.training.config import TrainConfig
from paper_replication.training.loop import train_model


@dataclass(frozen=True)
class HyperparameterCandidate:
    name: str
    model_cls: type[nn.Module]
    model_config: Any
    train_config: TrainConfig


@dataclass
class SearchResult:
    candidate: HyperparameterCandidate
    model: nn.Module
    history: list[dict[str, float]]
    best_epoch: dict[str, float]


def run_search(
    candidates: list[HyperparameterCandidate],
    train_dataset: AttnLOBDataset,
    val_dataset: AttnLOBDataset,
    on_result: Callable[[SearchResult], None] | None = None,
) -> list[SearchResult]:
    """Trains every candidate, returns results sorted best-val_loss-first.

    Each returned `SearchResult.model` already has its best-val_loss epoch's
    weights loaded (that's what `train_model` does internally) -- no
    retraining needed to use the winner.

    `on_result`, if given, is called with each `SearchResult` as soon as its
    candidate finishes training -- useful for progress reporting on a search
    that can take several minutes.
    """
    results = []
    for candidate in candidates:
        model = candidate.model_cls(candidate.model_config)
        num_classes: int = candidate.model_config.num_classes
        history = train_model(
            model, train_dataset, val_dataset, candidate.train_config, num_classes
        )
        best_epoch = min(history, key=lambda record: record["val_loss"])
        result = SearchResult(candidate, model, history, best_epoch)
        results.append(result)
        if on_result is not None:
            on_result(result)

    results.sort(key=lambda result: result.best_epoch["val_loss"])
    return results
