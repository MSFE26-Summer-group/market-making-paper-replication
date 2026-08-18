"""FC-LOB: the fully-connected MLP baseline from paper Table I / Section IV-B2.

"FC-LOB is a multi-layer perception network. The number of neurons in the
hidden layer is (1024, 256, 64, 3), and the activation function is leaky
Relu except for Softmax in the last layer."

Deviation from the paper: their FC-LOB uses a T=100 window (Input 4000x1 =
100*40 flattened). This implementation instead defaults to window_T=50,
matching AttnLOB and this replication's feature pipeline, so every baseline
in this replication is compared on the exact same dataset windows -- the
paper's own baselines already use inconsistent T per model (100 for
FC-LOB/DeepLOB, 1024 for Conv-LOB, 50 for Attn-LOB), so there's no single
"faithful" choice to match; apples-to-apples within this replication is the
more useful property here.

Returns raw logits from `forward()` (softmax happens inside
`nn.CrossEntropyLoss` during training), matching AttnLOB's convention.

Note on paper Table I's reported param count (256,064): chaining
`hidden_dims=(1024, 256, 64)` into a 3-class output, the last three layers
alone (256->64, 64->3, plus the 1024->256 in between) already total 279,043
params -- more than the paper's reported total for the *whole* network,
regardless of what feeds into the first layer. That's an internal
inconsistency in the paper's own numbers, not a fixable modeling choice, so
this implementation follows the stated layer widths literally rather than
reverse-engineering something that matches 256,064.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FCLOBConfig:
    """Hyperparameters for `FCLOB`.

    n_levels, window_T: together fix the flattened input size
        (window_T * 4 * n_levels) -- unlike AttnLOB's conv/attention stack,
        a plain MLP can't handle a variable T at inference time.
    hidden_dims: hidden layer widths (paper: 1024, 256, 64).
    num_classes: pretrain task classes ({down, stationary, up}, paper Eq. 5).
    """

    n_levels: int = 10
    window_T: int = 50
    hidden_dims: tuple[int, ...] = (1024, 256, 64)
    num_classes: int = 3

    @property
    def input_dim(self) -> int:
        return self.window_T * 4 * self.n_levels


class FCLOB(nn.Module):
    """MLP baseline (paper Table I / Section IV-B2)."""

    def __init__(self, config: FCLOBConfig | None = None) -> None:
        super().__init__()
        self.config = config or FCLOBConfig()

        dims = [self.config.input_dim, *self.config.hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.LeakyReLU()]
        layers.append(nn.Linear(dims[-1], self.config.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, lob_state: torch.Tensor) -> torch.Tensor:
        """lob_state: (batch, window_T, 4*n_levels) -> (batch, num_classes) logits."""
        batch_size = lob_state.shape[0]
        x = lob_state.reshape(batch_size, -1)
        if x.shape[1] != self.config.input_dim:
            expected_width = 4 * self.config.n_levels
            raise ValueError(
                f"Expected flattened input size {self.config.input_dim} "
                f"(window_T={self.config.window_T} * 4*n_levels={expected_width}), "
                f"got {x.shape[1]}. Unlike AttnLOB, FC-LOB needs a fixed window_T "
                f"matching its config."
            )
        logits: torch.Tensor = self.net(x)
        return logits
