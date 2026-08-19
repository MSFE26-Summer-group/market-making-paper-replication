"""Reward function components (paper III-D, Eq. 12-16)."""

from __future__ import annotations

from paper_replication.rl.simulator import Fill


def delta_pnl(value: float, previous_value: float) -> float:
    """Eq. 12: change in mark-to-market value over one step."""
    return value - previous_value


def dampened_pnl(pnl_delta: float, eta: float) -> float:
    """Eq. 13: dampens profit from holding inventory, leaves losses untouched."""
    return pnl_delta - max(0.0, eta * pnl_delta)


def trading_pnl(fills: list[Fill], mid_price: float) -> float:
    """Eq. 14: price advantage captured by this step's fills relative to mid-price.

    `Xt,v` (signed volume) is `fill.side * fill.volume`; a buy fill below
    mid-price or a sell fill above mid-price both contribute positively.
    """
    return sum(fill.side * fill.volume * (mid_price - fill.price) for fill in fills)


def inventory_punishment(inventory: float, zeta: float) -> float:
    """Eq. 15: quadratic penalty on inventory, so it grows quickly with size."""
    return zeta * inventory**2


def hybrid_reward(
    pnl_delta: float,
    fills: list[Fill],
    mid_price: float,
    inventory: float,
    eta: float,
    zeta: float,
) -> float:
    """Eq. 16: R_t = DP_t + TP_t - IP_t."""
    return (
        dampened_pnl(pnl_delta, eta)
        + trading_pnl(fills, mid_price)
        - inventory_punishment(inventory, zeta)
    )
