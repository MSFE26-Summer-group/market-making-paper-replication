"""Conv-LOB: the dilated fully-convolutional baseline from paper Table I / IV-B2.

"Conv-LOB is a fully convolutional network that uses dilated convolution to
accept longer sequences. The architecture is similar to [51]." [51] is
WaveNet (van den Oord et al., 2016) -- the paper doesn't specify block
count, kernel size, or whether WaveNet's gated activation/skip-connection
details are included, so this implementation is a standard causal dilated
residual stack (the same pattern popularized as "TCN"): each block is a
causal (left-padded only, so output at time t never sees t' > t) dilated
Conv1d with a doubling dilation schedule (1, 2, 4, 8, ...) and a residual
connection, no gating.

Unlike AttnLOB/DeepLOB, there's no width-collapsing stage first -- the
4*n_levels axis is treated as input channels directly, matching the "fully
convolutional" description (no dense layers, no flattening).

Block count (default 6, kernel_size=3) gives a receptive field of 127
timesteps -- comfortably larger than this replication's window_T=50, so the
final timestep's output has (causally) seen the entire window. Paper's own
Conv-LOB uses T=1024 (see the module docstring in `fc_lob.py` for why this
replication uses window_T=50 across all baselines instead).

Returns raw logits from `forward()`, matching the other models here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ConvLOBConfig:
    """Hyperparameters for `ConvLOB`.

    n_levels: number of LOB price levels (input channels = 4 * n_levels).
    hidden_channels: channels used throughout the dilated conv stack.
    kernel_size: causal conv kernel size, shared by every block.
    num_blocks: dilated residual blocks, dilation doubling per block
        (1, 2, 4, ...). Determines the receptive field -- see module
        docstring.
    num_classes: pretrain task classes ({down, stationary, up}, paper Eq. 5).
    """

    n_levels: int = 10
    hidden_channels: int = 32
    kernel_size: int = 3
    num_blocks: int = 6
    num_classes: int = 3

    @property
    def in_channels(self) -> int:
        return 4 * self.n_levels

    @property
    def receptive_field(self) -> int:
        added: int = sum(
            (self.kernel_size - 1) * (2**i) for i in range(self.num_blocks)
        )
        return 1 + added


class _CausalConv1d(nn.Module):
    """Conv1d that only looks backward in time: left-pads by (kernel-1)*dilation."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, T) -> (batch, channels, T), causally."""
        x = F.pad(x, (self.left_pad, 0))
        result: torch.Tensor = self.conv(x)
        return result


class _DilatedResidualBlock(nn.Module):
    """Causal dilated conv + LeakyReLU + residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv = _CausalConv1d(channels, kernel_size, dilation)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = x + self.activation(self.conv(x))
        return result


class ConvLOB(nn.Module):
    """Dilated fully-convolutional baseline (paper Table I / Section IV-B2)."""

    def __init__(self, config: ConvLOBConfig | None = None) -> None:
        super().__init__()
        self.config = config or ConvLOBConfig()

        self.input_proj = nn.Conv1d(
            self.config.in_channels, self.config.hidden_channels, 1
        )
        self.blocks = nn.Sequential(
            *[
                _DilatedResidualBlock(
                    self.config.hidden_channels, self.config.kernel_size, 2**i
                )
                for i in range(self.config.num_blocks)
            ]
        )
        self.head = nn.Linear(self.config.hidden_channels, self.config.num_classes)

    def forward(self, lob_state: torch.Tensor) -> torch.Tensor:
        """lob_state: (batch, T, 4*n_levels) -> (batch, num_classes) logits."""
        x = lob_state.permute(0, 2, 1)  # (batch, 4n, T)
        x = self.input_proj(x)  # (batch, hidden, T)
        x = self.blocks(x)  # (batch, hidden, T)
        # last timestep -> (batch, hidden); causal, so this saw the whole window
        pooled = x[:, :, -1]
        logits: torch.Tensor = self.head(pooled)
        return logits
