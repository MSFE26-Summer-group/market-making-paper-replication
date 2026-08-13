"""Attn-LOB: the CNN-Attention feature extractor from paper Section III-B3 / Fig. 1.

Architecture, in order:
1. A width-collapsing conv stack ("spatial dependencies" across the
   4*n_levels price/size columns), matching the paper's Fig. 1 kernel/stride
   sequence exactly for n_levels=10: (1x2, stride 1x2), (1x5, stride 1x5),
   (1x4) -- collapsing width 40 -> 20 -> 4 -> 1 while preserving the T axis.
2. An Inception module ("multi-scale temporal dependencies"): three
   parallel branches (1x1->3x1, 1x1->5x1, maxpool3x1->1x1) of 64 channels
   each, concatenated to 192 -- this follows DeepLOB ([23], the paper's own
   cited architecture source), whose Inception module is reused unchanged.
3. A multi-head self-attention layer over the T axis, replacing the LSTM
   used by DeepLOB/[19]/[23] -- this paper's stated contribution.

Only the pretrain head is implemented (Table I: raw LOB state only, input
shape 50x40, no dynamic or agent state). `forward_features` exposes the
pooled 192-dim backbone representation so a later RL-phase head can
concatenate dynamic/agent state onto it (paper Fig. 1's "Concatenate" box)
without needing to touch this module.

The paper's Fig. 1 is OCR'd from a scanned PDF and doesn't specify every
detail; where it's silent, this implementation makes a documented,
standard choice rather than guessing at unstated paper internals:
- Attention head count (default 4): not stated in the paper.
- Residual connection + LayerNorm around the attention layer: standard
  transformer-block practice, not explicitly shown in Fig. 1.
- Pooling the attention output at the *last* time step for a single
  192-dim vector: the natural analogue of "take the final hidden state"
  from the LSTM-based architecture this paper modifies, and matches the
  "1 x 192" shape Fig. 1 shows feeding the pretrain head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from paper_replication.models.layers import ConvBlock, InceptionModule


@dataclass(frozen=True)
class AttnLOBConfig:
    """Hyperparameters for `AttnLOB`.

    n_levels: number of LOB price levels; the conv stack's final layer
        collapses width 4*n_levels down to 1, which requires n_levels to be
        a multiple of 5 (see `layers.ConvBlock`). 10 is the paper's own
        setup (Fig. 1); other multiples of 5 are supported for depth
        ablations but deviate from the paper.
    conv_hidden_dim: channels after the width-collapsing conv stack.
    inception_channels: channels per Inception branch (3 branches -> embed_dim).
    attn_heads: multi-head self-attention head count.
    dropout: applied to the self-attention weights and to the pooled
        backbone representation before the pretrain head. Not part of the
        paper's Fig. 1 -- a regularization knob added after an untuned run
        showed val_loss overfitting within a handful of epochs. 0.0 (default)
        reproduces the original architecture exactly.
    num_classes: pretrain task classes ({down, stationary, up}, paper Eq. 5).
    """

    n_levels: int = 10
    conv_hidden_dim: int = 32
    inception_channels: int = 64
    attn_heads: int = 4
    dropout: float = 0.0
    num_classes: int = 3

    @property
    def embed_dim(self) -> int:
        return 3 * self.inception_channels


class AttnLOB(nn.Module):
    """CNN-Attention feature extractor + mid-price direction pretrain head.

    See module docstring for the full architecture description and the
    design choices made where the paper's Fig. 1 is underspecified.
    """

    def __init__(self, config: AttnLOBConfig | None = None) -> None:
        super().__init__()
        self.config = config or AttnLOBConfig()

        self.embed_dim = self.config.embed_dim

        self.conv_block = ConvBlock(self.config.conv_hidden_dim, self.config.n_levels)
        self.inception = InceptionModule(
            self.config.conv_hidden_dim, self.config.inception_channels
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.config.attn_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.embed_dim)
        self.head_dropout = nn.Dropout(self.config.dropout)
        self.pretrain_head = nn.Linear(self.embed_dim, self.config.num_classes)

    def forward_features(self, lob_state: torch.Tensor) -> torch.Tensor:
        """lob_state (batch, T, 4*n_levels) -> pooled (batch, embed_dim) backbone out.

        Attention output at the last time step, per the module docstring.
        """
        x = lob_state.unsqueeze(1)  # (batch, 1, T, 4n)
        x = self.conv_block(x)  # (batch, hidden_dim, T, 1)
        x = x.squeeze(-1)  # (batch, hidden_dim, T)
        x = self.inception(x)  # (batch, embed_dim, T)
        x = x.permute(0, 2, 1)  # (batch, T, embed_dim)

        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)  # residual + layernorm

        pooled: torch.Tensor = x[:, -1, :]  # last time step -> (batch, embed_dim)
        pooled = self.head_dropout(pooled)
        return pooled

    def forward(self, lob_state: torch.Tensor) -> torch.Tensor:
        """lob_state: (batch, T, 4*n_levels) -> (batch, num_classes) pretrain logits."""
        pooled = self.forward_features(lob_state)
        logits: torch.Tensor = self.pretrain_head(pooled)
        return logits
