"""DeepLOB: the CNN-LSTM baseline this paper builds on (Table I / [23]).

Zhang, Zohren & Roberts (2019), "DeepLOB: Deep Convolutional Neural
Networks for Limit Order Books" -- explicitly the architecture Attn-LOB
modifies: "They proposed a CNN-LSTM based network and we add a multi-head
self-attention layer to model temporal dependencies." This model reuses the
exact same conv stack (`layers.ConvBlock`) and Inception module
(`layers.InceptionModule`) as AttnLOB -- both share those two stages
verbatim -- and replaces AttnLOB's multi-head self-attention with an LSTM,
the one documented difference between the two.

Pooling: the LSTM's final hidden state (`h_n`, last layer) is used as the
pooled representation, the standard way to summarize an LSTM's output over
a sequence -- the direct analogue of AttnLOB's "last time step" pooling.

Returns raw logits from `forward()`, matching the other models here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from paper_replication.models.layers import ConvBlock, InceptionModule


@dataclass(frozen=True)
class DeepLOBConfig:
    """Hyperparameters for `DeepLOB`.

    n_levels: number of LOB price levels; the conv stack's final layer
        collapses width 4*n_levels down to 1, which requires n_levels to be
        a multiple of 5 (see `layers.ConvBlock`, shared with AttnLOB). 10 is
        the paper's own setup (Fig. 1).
    conv_hidden_dim: channels after the width-collapsing conv stack.
    inception_channels: channels per Inception branch (3 branches -> LSTM input).
    lstm_hidden_dim: LSTM hidden size.
    num_classes: pretrain task classes ({down, stationary, up}, paper Eq. 5).
    """

    n_levels: int = 10
    conv_hidden_dim: int = 32
    inception_channels: int = 64
    lstm_hidden_dim: int = 64
    num_classes: int = 3

    @property
    def lstm_input_dim(self) -> int:
        return 3 * self.inception_channels


class DeepLOB(nn.Module):
    """CNN-LSTM feature extractor + mid-price direction pretrain head."""

    def __init__(self, config: DeepLOBConfig | None = None) -> None:
        super().__init__()
        self.config = config or DeepLOBConfig()

        self.conv_block = ConvBlock(self.config.conv_hidden_dim, self.config.n_levels)
        self.inception = InceptionModule(
            self.config.conv_hidden_dim, self.config.inception_channels
        )
        self.lstm = nn.LSTM(
            input_size=self.config.lstm_input_dim,
            hidden_size=self.config.lstm_hidden_dim,
            batch_first=True,
        )
        self.pretrain_head = nn.Linear(
            self.config.lstm_hidden_dim, self.config.num_classes
        )

    def forward_features(self, lob_state: torch.Tensor) -> torch.Tensor:
        """lob_state (batch, T, 4*n_levels) -> pooled (batch, lstm_hidden_dim)."""
        x = lob_state.unsqueeze(1)  # (batch, 1, T, 4n)
        x = self.conv_block(x)  # (batch, hidden_dim, T, 1)
        x = x.squeeze(-1)  # (batch, hidden_dim, T)
        x = self.inception(x)  # (batch, lstm_input_dim, T)
        x = x.permute(0, 2, 1)  # (batch, T, lstm_input_dim)

        _, (h_n, _) = self.lstm(x)
        # final layer's last hidden state -> (batch, hidden)
        pooled: torch.Tensor = h_n[-1]
        return pooled

    def forward(self, lob_state: torch.Tensor) -> torch.Tensor:
        """lob_state: (batch, T, 4*n_levels) -> (batch, num_classes) pretrain logits."""
        pooled = self.forward_features(lob_state)
        logits: torch.Tensor = self.pretrain_head(pooled)
        return logits
