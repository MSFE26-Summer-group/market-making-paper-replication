"""Building blocks shared by AttnLOB and DeepLOB (paper Fig. 1 / DeepLOB [23]).

Both models' first two stages -- the width-collapsing conv stack and the
Inception module -- are identical; they only diverge at the final temporal
aggregation step (AttnLOB: multi-head self-attention; DeepLOB: LSTM). These
two classes are defined once here rather than duplicated per model.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Collapses the 4*n_levels feature axis to `hidden_dim` channels; T unchanged.

    Kernel/stride sequence is (1x2, stride 1x2), (1x5, stride 1x5), then a
    final (1xK) collapse where K = 4*n_levels/10 -- for n_levels=10 that's
    (1x2, 1x5, 1x4), matching paper Fig. 1 exactly. Requires 4*n_levels to
    be divisible by 10 (i.e. n_levels a multiple of 5) so the final layer
    collapses width to exactly 1.
    """

    def __init__(self, hidden_dim: int, n_levels: int = 10) -> None:
        super().__init__()
        width = 4 * n_levels
        if width % 10 != 0:
            raise ValueError(
                f"ConvBlock's (1x2, 1x5, 1xK) collapse needs n_levels to be a "
                f"multiple of 5 -- got n_levels={n_levels} (4*n_levels={width})"
            )
        final_kernel = width // 10
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 5), stride=(1, 5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, final_kernel)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, T, 4*n_levels) -> (batch, hidden_dim, T, 1)."""
        result: torch.Tensor = self.net(x)
        return result


class InceptionModule(nn.Module):
    """Multi-scale temporal conv (Fig. 1 / DeepLOB [23]): 3 branches, concatenated."""

    def __init__(self, in_channels: int, branch_channels: int) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_channels, T) -> (batch, 3*branch_channels, T)."""
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
