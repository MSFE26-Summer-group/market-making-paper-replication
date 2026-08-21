"""Market simulator: executes agent quotes against historical best bid/ask.

Paper reference: Section IV-C1.

The paper's simulator is event-driven: "the simulator executes the agent's
order only when the real historical order arrives" -- an agent's resting
limit order fills, at the agent's own quoted price, the moment a real
historical order crosses it. Our data is a fixed ~10s snapshot grid rather
than a raw order-event log (see `docs/feature_engineering.md`'s divergence
notes), so there's no per-order timeline to replay. This module adapts the
same rule to the snapshot grid: an agent's resting quote fills, at the
agent's own price, if the *next* snapshot's opposing best price has crossed
it -- the snapshot-grid analogue of "a real order arrived that would have
matched this quote." Volume is always exactly `minimum_trade_unit` (paper
III-C1: "the volume of each order is the minimum trade unit").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fill:
    """One executed trade.

    side: +1 buy, -1 sell -- matches the paper's `Xt,v` signed-volume
        convention (Eq. 14): positive for buying, negative for selling.
    """

    side: int
    price: float
    volume: float


@dataclass
class MarketSimulator:
    """Tracks an agent's cash/inventory and executes quotes against a snapshot grid.

    minimum_trade_unit: volume of every quoted (non-close) fill.
    max_inventory: `omega * minimum_trade_unit` (paper III-C2) -- a quote
        that would push inventory past this in either direction is silently
        dropped, mirroring "the agent will be prohibited from placing orders
        in that direction."
    """

    minimum_trade_unit: float
    max_inventory: float
    cash: float = 0.0
    inventory: float = 0.0

    def reset(self, cash: float = 0.0) -> None:
        self.cash = cash
        self.inventory = 0.0

    def value(self, mid_price: float) -> float:
        """Mark-to-market value: cash + inventory * mid_price (paper Eq. 12)."""
        return self.cash + self.inventory * mid_price

    def _apply(self, side: int, price: float, volume: float) -> Fill:
        self.cash -= side * price * volume
        self.inventory += side * volume
        return Fill(side=side, price=price, volume=volume)

    def close_position(self, next_best_bid: float, next_best_ask: float) -> list[Fill]:
        """Closes any open position with a market order (paper III-C1, IV-C2).

        Executed "at the counterparty's price": a long position sells at the
        market's best bid, a short position buys at the market's best ask.
        """
        if self.inventory > 0:
            return [self._apply(-1, next_best_bid, self.inventory)]
        if self.inventory < 0:
            return [self._apply(1, next_best_ask, -self.inventory)]
        return []

    def step(
        self,
        bid_price: float | None,
        ask_price: float | None,
        next_best_bid: float,
        next_best_ask: float,
    ) -> list[Fill]:
        """Fills any of the agent's resting quotes that the next snapshot crosses.

        `bid_price`/`ask_price` is `None` to skip quoting on that side.
        """
        fills: list[Fill] = []
        if (
            bid_price is not None
            and self.inventory + self.minimum_trade_unit <= self.max_inventory
            and bid_price >= next_best_ask
        ):
            fills.append(self._apply(1, bid_price, self.minimum_trade_unit))
        if (
            ask_price is not None
            and self.inventory - self.minimum_trade_unit >= -self.max_inventory
            and ask_price <= next_best_bid
        ):
            fills.append(self._apply(-1, ask_price, self.minimum_trade_unit))
        return fills
