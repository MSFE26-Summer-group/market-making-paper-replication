"""torch.utils.data.Dataset wrapper around paper_replication.features.AttnLOBDataset."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from paper_replication.features.dataset import AttnLOBDataset


class AttnLOBTorchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Yields (lob_state, label_index) tensor pairs for `AttnLOB`.

    Labels are stored as {-1, 0, 1} (down/stationary/up) in `AttnLOBDataset`
    but `nn.CrossEntropyLoss` needs class indices >= 0, so they're remapped
    here to {0, 1, 2} via `label + 1` -- index 0=down, 1=stationary, 2=up,
    matching `AttnLOBConfig.num_classes=3`'s output ordering.
    """

    def __init__(self, dataset: AttnLOBDataset) -> None:
        self.lob_state = torch.from_numpy(dataset.lob_state).float()
        self.labels = torch.from_numpy(dataset.labels.astype(np.int64) + 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lob_state[index], self.labels[index]
