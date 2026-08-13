"""Tests for paper_replication.training.metrics."""

import pytest
import torch

from paper_replication.training.metrics import confusion_matrix, precision_recall_f1


def test_confusion_matrix_matches_hand_count() -> None:
    y_true = torch.tensor([0, 0, 1, 1, 2, 2])
    y_pred = torch.tensor([0, 1, 1, 1, 2, 0])

    cm = confusion_matrix(y_true, y_pred, num_classes=3)

    expected = torch.tensor(
        [
            [1, 1, 0],  # true=0: one predicted 0, one predicted 1
            [0, 2, 0],  # true=1: both predicted 1
            [1, 0, 1],  # true=2: one predicted 0, one predicted 2
        ]
    )
    torch.testing.assert_close(cm, expected)


def test_precision_recall_f1_matches_hand_calculation() -> None:
    confusion = torch.tensor([[1, 1, 0], [0, 2, 0], [1, 0, 1]])

    metrics = precision_recall_f1(confusion)

    # class0: precision=1/2, recall=1/2 | class1: precision=2/3, recall=2/2
    # class2: precision=1/1, recall=1/2
    expected_precision = (0.5 + 2 / 3 + 1.0) / 3
    expected_recall = (0.5 + 1.0 + 0.5) / 3
    expected_accuracy = 4 / 6

    assert metrics["precision"] == pytest.approx(expected_precision)
    assert metrics["recall"] == pytest.approx(expected_recall)
    assert metrics["accuracy"] == pytest.approx(expected_accuracy)


def test_precision_recall_f1_handles_empty_classes_without_division_by_zero() -> None:
    # class 1 never predicted, class 2 never true
    confusion = torch.tensor([[2, 0, 0], [1, 0, 0], [0, 0, 0]])

    metrics = precision_recall_f1(confusion)

    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_perfect_predictions_give_all_ones() -> None:
    confusion = torch.tensor([[3, 0, 0], [0, 4, 0], [0, 0, 2]])

    metrics = precision_recall_f1(confusion)

    assert metrics == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0}
