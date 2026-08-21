"""Configuration for the market-making RL environment (paper III-C, III-D, IV-C2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RLConfig:
    """Hyperparameters for `MarketMakingEnv` and its simulator/reward/action space.

    Field names and defaults follow the paper's own notation and reported
    values (Section IV-C2: `omega=10`, `eta=0.5`, `zeta=0.01`, `max_bias=0.05`,
    `max_spread=0.1`) except where the paper's own units don't transfer to
    this dataset -- see each field below for what diverges and why.

    minimum_trade_unit: volume of every non-close fill (paper III-C1: "the
        volume of each order is the minimum trade unit, which is 100 in the
        China Stock Market"). 100 shares doesn't mean anything for BTC/USDT,
        so this defaults to 0.001 BTC instead -- a documented substitute for
        this asset, not the paper's own number.
    omega: `max_inventory = omega * minimum_trade_unit` (paper III-C2):
        quoting in a direction that would push inventory past this is
        prohibited.
    initial_cash: value the simulator's cash resets to at the start of every
        episode (paper IV-C2: "the agent's value is initialized to 0").
    max_bias: paper's `max_bias` (Eq. 8) -- maximum distance between the
        reservation price and mid-price.
    max_spread: paper's `max_spread` (Eq. 10) -- maximum quoted spread.
    eta: paper's `eta` (Eq. 13) -- Dampened PnL's asymmetric punishment
        strength.
    zeta: paper's `zeta` (Eq. 15) -- Inventory Punishment's quadratic
        coefficient.
    episode_length: steps per episode (paper IV-C2: 2000 raw LOB events,
        "about 3-5 minutes"). Our data is a fixed ~10s snapshot grid rather
        than an event log (see `docs/feature_engineering.md`'s divergence
        notes), so reusing "2000" would span days, not minutes. This default
        (30 steps ~= 5 minutes at ~10s/step) preserves the paper's
        wall-clock episode length instead of its row count -- the opposite
        trade-off the pretraining pipeline made for `window_T`/`horizon_k`,
        and a deliberate choice: episode length determines how much market
        history a single trajectory spans, which is a wall-clock concept.
    """

    minimum_trade_unit: float = 0.001
    omega: float = 10.0
    initial_cash: float = 0.0

    max_bias: float = 0.05
    max_spread: float = 0.1

    eta: float = 0.5
    zeta: float = 0.01

    episode_length: int = 30

    @property
    def max_inventory(self) -> float:
        return self.omega * self.minimum_trade_unit
