"""Precomputes the frozen Attn-LOB backbone's output over an entire `MarketDataset`.

RL training keeps the Attn-LOB backbone frozen at its pretrained weights
(paper Fig. 1's own pretrain-then-RL structure) and only updates the small
trunk + policy/value heads (see `networks.py`'s module docstring). Since the
backbone's output doesn't depend on the agent's actions -- only
`state.agent_state_vector`'s inventory feature does -- it can be computed
once, in large batches, instead of once per RL step. This turns the RL
loop's dominant cost from "one CNN+attention forward pass per step" into
"one array lookup per step," which is what makes training tractable here.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import torch

from paper_replication.models.attn_lob import AttnLOB

FloatArray: TypeAlias = npt.NDArray[np.float64]
Float32Array: TypeAlias = npt.NDArray[np.float32]


@torch.no_grad()
def precompute_lob_features(
    attn_lob: AttnLOB,
    lob_state: FloatArray,
    device: str = "cpu",
    batch_size: int = 1024,
) -> Float32Array:
    """Runs `AttnLOB.forward_features` over `lob_state` in batches.

    `(n_steps, T, 4*n_levels) -> (n_steps, embed_dim)`. Runs in `eval()`
    mode (dropout off) so the cached features are deterministic.
    `attn_lob`'s own device/mode are restored before returning.
    """
    was_training = attn_lob.training
    original_device = next(attn_lob.parameters()).device
    attn_lob.to(device)
    attn_lob.eval()

    outputs = []
    try:
        for start in range(0, len(lob_state), batch_size):
            batch = torch.from_numpy(lob_state[start : start + batch_size]).float()
            batch = batch.to(device)
            features = attn_lob.forward_features(batch)
            outputs.append(features.cpu().numpy())
    finally:
        attn_lob.to(original_device)
        attn_lob.train(was_training)

    result: Float32Array = np.concatenate(outputs, axis=0).astype(np.float32)
    return result
