"""Tests for paper_replication.training.dataset."""

import numpy as np
import torch

from paper_replication.features.dataset import AttnLOBDataset
from paper_replication.training.dataset import AttnLOBTorchDataset


def _make_dataset(n: int = 5) -> AttnLOBDataset:
    return AttnLOBDataset(
        lob_state=np.random.default_rng(0).normal(size=(n, 50, 40)),
        dynamic_state=np.zeros((n, 1)),
        dynamic_feature_names=["dummy"],
        labels=np.array([-1, 0, 1, -1, 0][:n], dtype=np.int8),
        timestamps=np.arange(n, dtype=np.float64),
        alpha_used=0.0,
    )


def test_length_matches_underlying_dataset() -> None:
    torch_dataset = AttnLOBTorchDataset(_make_dataset(5))
    assert len(torch_dataset) == 5


def test_item_shapes_and_dtypes() -> None:
    torch_dataset = AttnLOBTorchDataset(_make_dataset(5))

    x, y = torch_dataset[0]

    assert x.shape == (50, 40)
    assert x.dtype == torch.float32
    assert y.shape == ()
    assert y.dtype == torch.int64


def test_labels_remapped_from_signed_to_class_index() -> None:
    torch_dataset = AttnLOBTorchDataset(_make_dataset(5))

    labels = [int(torch_dataset[i][1]) for i in range(5)]

    # underlying labels [-1, 0, 1, -1, 0] -> class indices [0, 1, 2, 0, 1]
    assert labels == [0, 1, 2, 0, 1]
