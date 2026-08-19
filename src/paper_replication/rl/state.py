"""Agent state: inventory and remaining-time features fed to the policy.

Paper reference: Section III-A3.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]

N_AGENT_FEATURES = 2


def agent_state_vector(
    inventory: float, max_inventory: float, elapsed_frac: float
) -> FloatArray:
    """(2,) vector: inventory normalized by `max_inventory`, and elapsed time fraction.

    The paper defines "Remaining Time" as current time `t` divided by total
    time `T` (III-A3) -- despite the name, that's the *elapsed* fraction of
    the episode, not time remaining. We keep the paper's own formula rather
    than "fixing" it to match what the name implies.
    """
    normalized_inventory = inventory / max_inventory if max_inventory != 0 else 0.0
    return np.array([normalized_inventory, elapsed_frac], dtype=np.float64)
