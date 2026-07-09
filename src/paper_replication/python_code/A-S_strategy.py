from typing import Dict

import numpy.log as log

class AvellanedaStoikovAgent:
    def __init__(
        self,
        gamma: float,
        sigma: float,
        kappa: float,
        A: float,
        T_session: float,
        inventory: float,
        cash: float = 100000,
        reservation_price: float | None = None,
        current_bid: float | None = None,
        current_ask: float | None = None,
    ):
        # risk aversion
        self.gamma = gamma
        # volatility per-period (consistent with T)
        self.sigma = sigma
        # Order book liquidity density
        self.kappa = kappa
        # Order arrival intensity
        self.A = A
        # Total session duration
        self.T_session = T_session

        # State vars
        self.inventory = inventory
        self.cash = cash

        # Market vars (for observation and debugging
        self.reservation_price = reservation_price
        self.current_bid = current_bid
        self.current_ask = current_ask

    def get_quotes(
        self, mid_price: float, current_time: float
    ) -> Dict[str, float | None]:
        # Remaining time
        T_left = max(0, self.T_session - current_time)

        # Calculate reservation price
        # r = s - q * gamma * sigma^2 * (T - t)
        # adjust the "fair value" based on inventory risk.
        # if q > 0 (long), r < s we lower price to sell.
        self.reservation_price = mid_price - (
            self.inventory * self.gamma * (self.sigma**2) * T_left
        )

        # calculate Optimal Spread (delta)
        # delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/kappa)
        # the first term is the risk component, the second is the liquidity component.
        risk_component = self.gamma * (self.sigma**2) * T_left
        liquidity_component = (2 / self.gamma) * log(1 + (self.gamma / self.kappa))

        optimal_spread = risk_component + liquidity_component
        half_spread = optimal_spread / 2

        # set Bid/Ask around Reservation Price
        self.current_bid = self.reservation_price - half_spread
        self.current_ask = self.reservation_price + half_spread

        return {"bid": self.current_bid, "ask": self.current_ask}

    # execute a trade
    # quantity is usually 1 or a standard lot size in simulations
    # side: 1 = Agent BUY, -1 = Agent SELL
    def fill_order(self, price: float, quantity: float, side: int) -> None:
        if side == 1:
            self.inventory = self.inventory + quantity
            self.cash = self.cash - (price * quantity)
        else:
            self.inventory = self.inventory - quantity
            self.cash = self.cash + (price * quantity)

    # Mark-to-Market wealth
    def get_wealth(self, mid_price: float) -> float:
        return self.cash + self.inventory * mid_price

