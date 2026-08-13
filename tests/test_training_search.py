"""Tests for paper_replication.training.search."""

import numpy as np
import torch

from paper_replication.features.dataset import AttnLOBDataset
from paper_replication.models.attn_lob import AttnLOB, AttnLOBConfig
from paper_replication.training.config import TrainConfig
from paper_replication.training.search import HyperparameterCandidate, run_search


def _trivial_dataset(n_per_class: int = 15, seed: int = 0) -> AttnLOBDataset:
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


def _candidate(name: str, learning_rate: float) -> HyperparameterCandidate:
    return HyperparameterCandidate(
        name=name,
        model_cls=AttnLOB,
        model_config=AttnLOBConfig(
            conv_hidden_dim=8, inception_channels=8, attn_heads=2
        ),
        train_config=TrainConfig(
            batch_size=16,
            epochs=8,
            patience=None,
            learning_rate=learning_rate,
            device="cpu",
            seed=0,
        ),
    )


def test_run_search_returns_one_result_per_candidate() -> None:
    dataset = _trivial_dataset()
    candidates = [_candidate("fast", 5e-3), _candidate("slow", 1e-4)]

    results = run_search(candidates, dataset, dataset)

    assert len(results) == 2
    assert {r.candidate.name for r in results} == {"fast", "slow"}


def test_run_search_sorts_by_best_val_loss_ascending() -> None:
    dataset = _trivial_dataset()
    # a near-zero and a very tiny learning rate should be clearly separable:
    # the tiny one barely moves off its random initialization in 8 epochs.
    candidates = [_candidate("well_tuned", 5e-3), _candidate("barely_moves", 1e-6)]

    results = run_search(candidates, dataset, dataset)

    val_losses = [r.best_epoch["val_loss"] for r in results]
    assert val_losses == sorted(val_losses)
    assert results[0].candidate.name == "well_tuned"


def test_on_result_callback_fires_once_per_candidate_in_run_order() -> None:
    dataset = _trivial_dataset()
    candidates = [_candidate("first", 5e-3), _candidate("second", 1e-6)]
    seen_names = []

    run_search(
        candidates,
        dataset,
        dataset,
        on_result=lambda r: seen_names.append(r.candidate.name),
    )

    # callback order follows training order (as candidates were given), not
    # the final sorted-by-val_loss order of the returned list.
    assert seen_names == ["first", "second"]


def test_best_epoch_matches_minimum_in_history() -> None:
    dataset = _trivial_dataset()
    results = run_search([_candidate("only", 5e-3)], dataset, dataset)

    result = results[0]
    expected = min(result.history, key=lambda record: record["val_loss"])

    assert result.best_epoch == expected


def test_returned_model_has_best_epoch_weights_loaded() -> None:
    dataset = _trivial_dataset()
    results = run_search([_candidate("only", 5e-3)], dataset, dataset)
    result = results[0]

    result.model.eval()
    with torch.no_grad():
        x = torch.from_numpy(dataset.lob_state).float()
        y = torch.from_numpy(dataset.labels.astype(np.int64) + 1)
        actual_loss = torch.nn.functional.cross_entropy(result.model(x), y).item()

    assert actual_loss == result.best_epoch["val_loss"] or actual_loss <= (
        result.best_epoch["val_loss"] + 1e-4
    )
