"""Baseline quoting strategies for comparison against C-PPO / D-DQN (paper IV-C3).

Every baseline below emits a continuous-space action `(a1, a2) in [0, 1]^2`
(paper Eq. 8-11) so it runs through the exact same `MarketMakingEnv`,
`ContinuousActionSpace`, `MarketSimulator`, and reward/metrics pipeline as
C-PPO -- no separate execution path, no separate metrics code.

The paper's own Random/Fixed baselines quote at specific *price levels* of
the order book ("randomly quotes ... in the five levels", "constantly
quotes ... in the fixed (1-3) level"). This dataset's `MarketDataset` only
carries level-1 (best bid/ask) prices -- the simulator never needed deeper
levels (see `simulator.py`'s divergence notes) -- so Random/Fixed are
reimplemented here as random/fixed *spread fractions* around the mid-price
instead of book levels. This is a genuine substitution, not the paper's own
definition, and is documented rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from paper_replication.rl.env import Observation

FloatArray: TypeAlias = npt.NDArray[np.float64]
ContinuousAction: TypeAlias = npt.NDArray[np.float64]


class RandomPolicy:
    """Quotes a uniformly random bias/spread each step.

    Paper: "Random Quoting Strategy".
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(self, observation: Observation, row: int) -> ContinuousAction:
        action: ContinuousAction = self._rng.uniform(0.0, 1.0, size=2)
        return action


@dataclass(frozen=True)
class FixedSpreadPolicy:
    """Quotes a constant, centered spread every step (paper: "Fixed Quoting Strategy").

    `spread_frac` (a fraction of `RLConfig.max_spread`) stands in for the
    paper's fixed *level* -- a narrower `spread_frac` is this replication's
    substitute for a tighter, closer-to-mid-price fixed level.
    """

    spread_frac: float

    def __call__(self, observation: Observation, row: int) -> ContinuousAction:
        return np.array([0.0, self.spread_frac])


@dataclass(frozen=True)
class AvellanedaStoikovPolicy:
    """Avellaneda-Stoikov-inspired baseline (paper Eq. 17-18), adapted to this
    replication's continuous action space so it runs through the same env
    pipeline as every other policy compared here.

    Reservation price / optimal spread follow Eq. 17-18 exactly, with
    `T_session` normalized to 1 (matching `python_code/A-S_strategy.py`'s
    own convention) so `T - t` is just `1 - elapsed_frac`, read straight off
    `observation.agent_state[1]`. `sigma` is read per-step from the
    dataset's own realized-volatility dynamic feature (`rv_feature_index`)
    rather than calibrated once for the whole run, so it tracks the
    market's actual volatility regime. `gamma` (risk aversion) and `kappa`
    (liquidity) are fixed constants here, not calibrated from order-arrival
    data -- this replication doesn't implement the paper's own `A`/`kappa`
    estimation -- a documented simplification, not the paper's calibrated
    AS baseline.

    The computed reservation price and spread are converted back into
    `(a1, a2)` (`|reservation_price - mid_price| / max_bias`,
    `spread / max_spread`) rather than quoted as raw prices directly:
    `ContinuousActionSpace.decode` reapplies the sign via
    `Sign(inventory)` (Eq. 9), which already matches this formula's own
    inventory-skew direction, so only the magnitude needs to be handed over.
    """

    mid_price: FloatArray
    max_inventory: float
    max_bias: float
    max_spread: float
    rv_feature_index: int
    gamma: float = 0.1
    kappa: float = 1.5

    def __call__(self, observation: Observation, row: int) -> ContinuousAction:
        mid_price = self.mid_price[row]
        inventory = observation.agent_state[0] * self.max_inventory
        elapsed_frac = observation.agent_state[1]
        t_left = max(0.0, 1.0 - elapsed_frac)
        sigma = max(float(observation.dynamic_state[self.rv_feature_index]), 1e-8)

        reservation_price = mid_price - inventory * self.gamma * sigma**2 * t_left
        risk_component = self.gamma * sigma**2 * t_left
        liquidity_component = (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
        spread = risk_component + liquidity_component

        bias = abs(reservation_price - mid_price)
        a1 = (
            float(np.clip(bias / self.max_bias, 0.0, 1.0)) if self.max_bias > 0 else 0.0
        )
        a2 = (
            float(np.clip(spread / self.max_spread, 0.0, 1.0))
            if self.max_spread > 0
            else 0.0
        )
        return np.array([a1, a2])
