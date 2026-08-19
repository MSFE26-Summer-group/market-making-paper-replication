"""Action spaces: turn the agent's raw action into a pair of quotes (paper III-C).

Both spaces restrict the agent to a single resting buy and a single resting
sell order -- it cannot otherwise exit the market (paper III-C, following
Chakraborty and Kearns [50]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Quote:
    """One step's desired quotes. `close_position=True` ignores bid/ask_price."""

    bid_price: float | None
    ask_price: float | None
    close_position: bool = False


def _reservation_price(mid_price: float, inventory: float, delta: float) -> float:
    """Paper Eq. 9: bias the reservation price away from mid-price toward
    reducing inventory risk -- buy-side bias when short, sell-side when long.
    """
    sign = float(np.sign(inventory))
    return mid_price - sign * delta


@dataclass(frozen=True)
class ContinuousActionSpace:
    """Paper III-C2 (Eq. 8-11): 2 actions in [0, 1] -> reservation price + spread.

    `A1` controls the bias `delta` of the reservation price `pr` away from
    the mid-price `pm` (Eq. 8-9); `A2` controls the quoted spread around
    `pr` (Eq. 10-11).
    """

    max_bias: float
    max_spread: float
    action_dim: int = 2

    def decode(
        self, action: npt.NDArray[np.float64], mid_price: float, inventory: float
    ) -> Quote:
        a1 = float(np.clip(action[0], 0.0, 1.0))
        a2 = float(np.clip(action[1], 0.0, 1.0))

        delta = a1 * self.max_bias  # Eq. 8
        reservation_price = _reservation_price(mid_price, inventory, delta)

        spread = a2 * self.max_spread  # Eq. 10
        bid_price = reservation_price - spread / 2  # Eq. 11
        ask_price = reservation_price + spread / 2
        return Quote(bid_price=bid_price, ask_price=ask_price)


@dataclass(frozen=True)
class DiscreteActionSpace:
    """Paper III-C1: 8 actions, "a pair of orders with a particular spread and
    bias"; action 7 closes the position with a market order.

    The paper doesn't enumerate the 7 quoting actions' exact spread/bias
    levels, so this fills them in with a documented, monotonically
    increasing grid over the same `(spread, bias)` parameters the continuous
    space uses (Eq. 8-11), rather than guessing at unstated paper internals:
    3 spread levels (as fractions of `max_spread`) crossed with "centered"
    (bias=0) vs. "skewed toward reducing inventory" (bias=`max_bias`), plus
    one extra narrow/skewed action to fill the 7th quoting slot.
    """

    max_bias: float
    max_spread: float
    n_actions: int = 8
    levels: tuple[tuple[float, float], ...] = (
        (0.25, 0.0),
        (0.25, 1.0),
        (0.5, 0.0),
        (0.5, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.1, 1.0),
    )  # (spread_frac, bias_frac) for actions 0-6; action 7 (index n_actions-1) = close

    def decode(self, action: int, mid_price: float, inventory: float) -> Quote:
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action}")
        if action == self.n_actions - 1:
            return Quote(bid_price=None, ask_price=None, close_position=True)

        spread_frac, bias_frac = self.levels[action]
        delta = bias_frac * self.max_bias
        reservation_price = _reservation_price(mid_price, inventory, delta)

        spread = spread_frac * self.max_spread
        bid_price = reservation_price - spread / 2
        ask_price = reservation_price + spread / 2
        return Quote(bid_price=bid_price, ask_price=ask_price)
