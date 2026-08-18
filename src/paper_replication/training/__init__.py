"""Model-agnostic training loop for the pretrain classification task (paper IV-B)."""

from paper_replication.training.checkpoint import load_checkpoint, save_checkpoint
from paper_replication.training.config import TrainConfig
from paper_replication.training.dataset import AttnLOBTorchDataset
from paper_replication.training.loop import evaluate, train_model, train_one_epoch
from paper_replication.training.metrics import confusion_matrix, precision_recall_f1
from paper_replication.training.search import (
    HyperparameterCandidate,
    SearchResult,
    run_search,
)

__all__ = [
    "TrainConfig",
    "AttnLOBTorchDataset",
    "train_model",
    "train_one_epoch",
    "evaluate",
    "confusion_matrix",
    "precision_recall_f1",
    "save_checkpoint",
    "load_checkpoint",
    "HyperparameterCandidate",
    "SearchResult",
    "run_search",
]
