"""Market-making environment over historical LOB snapshots.

Gym-style episodic environment following Guo, Lin & Huang (2023):

- Action: continuous (reservation-price bias, half-spread), both in
  basis points of the current mid price.
- Fill model: the agent's quotes rest for one 10s interval; a quote is
  filled when the interval's traded price range crosses it
  (min_trade_price <= bid -> buy fill, max_trade_price >= ask -> sell
  fill). One unit per fill, no market impact (historical replay).
- Reward: trading PnL + dampened holding PnL - inventory penalty.
- Inventory is liquidated at mid at the end of each episode.
"""

from dataclasses import dataclass, field

import numpy as np

from paper_replication.rl.data import LOBDataset


@dataclass
class EnvConfig:
    episode_len: int = 360  # 360 x 10s = 1 hour
    max_bias_bps: float = 5.0  # |reservation bias| cap
    min_half_spread_bps: float = 0.5
    max_half_spread_bps: float = 10.0
    dampen: float = 0.7  # fraction of positive holding PnL removed
    inv_penalty: float = 0.01  # lambda * q^2, in bps-of-mid units
    max_inventory: int = 10  # hard position cap


@dataclass
class StepResult:
    state: np.ndarray
    aux: np.ndarray  # (inventory / max_inventory, lob_imbalance)
    reward: float
    done: bool
    info: dict[str, float] = field(default_factory=dict)


class MarketMakingEnv:
    """Historical-replay market-making environment."""

    def __init__(
        self,
        data: LOBDataset,
        config: EnvConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.data = data
        self.cfg = config or EnvConfig()
        self.rng = rng or np.random.default_rng(0)
        self.t = 0
        self.start = 0
        self.inventory = 0
        self.cash = 0.0

    def _aux(self) -> np.ndarray:
        return np.array(
            [
                self.inventory / self.cfg.max_inventory,
                self.data.imbalance[self.t],
            ],
            dtype=np.float32,
        )

    def reset(self, start: int | None = None) -> StepResult:
        """Start a new episode at ``start`` (random if omitted)."""
        last_valid = len(self.data.mid) - self.cfg.episode_len - 1
        self.start = int(self.rng.integers(0, last_valid)) if start is None else start
        self.t = self.start
        self.inventory = 0
        self.cash = 0.0
        return StepResult(
            state=self.data.states[self.t], aux=self._aux(), reward=0.0, done=False
        )

    def step(self, action: np.ndarray) -> StepResult:
        """Quote around mid for one interval; action in [-1, 1]^2."""
        cfg = self.cfg
        mid = self.data.mid[self.t]

        bias_bps = float(np.clip(action[0], -1, 1)) * cfg.max_bias_bps
        half_bps = cfg.min_half_spread_bps + (
            float(np.clip(action[1], -1, 1)) + 1
        ) / 2 * (cfg.max_half_spread_bps - cfg.min_half_spread_bps)

        center = mid * (1 + bias_bps * 1e-4)
        bid = center * (1 - half_bps * 1e-4)
        ask = center * (1 + half_bps * 1e-4)

        # Fills happen during the interval (t, t+1].
        nxt = self.t + 1
        mid_next = self.data.mid[nxt]
        trading_pnl = 0.0

        buy_fill = (
            self.data.trade_min[nxt] <= bid and self.inventory < cfg.max_inventory
        )
        sell_fill = (
            self.data.trade_max[nxt] >= ask and self.inventory > -cfg.max_inventory
        )
        if buy_fill:
            self.inventory += 1
            self.cash -= bid
            trading_pnl += mid_next - bid
        if sell_fill:
            self.inventory -= 1
            self.cash += ask
            trading_pnl += ask - mid_next

        # Holding PnL on the position carried into the interval, with the
        # speculative (positive) part dampened.
        carried = self.inventory - int(buy_fill) + int(sell_fill)
        holding = carried * (mid_next - mid)
        if holding > 0:
            holding *= 1 - cfg.dampen

        penalty = cfg.inv_penalty * (self.inventory**2) * mid * 1e-4

        # Rewards are scaled to bps of mid so they are size-independent.
        scale = 1e4 / mid
        reward = (trading_pnl + holding) * scale - penalty * scale

        self.t = nxt
        done = self.t - self.start >= cfg.episode_len
        info = {
            "bid": bid,
            "ask": ask,
            "inventory": float(self.inventory),
            "trading_pnl": trading_pnl,
            "nav": self.cash + self.inventory * mid_next,
        }
        if done:  # liquidate at mid, as in the paper's episode reset
            self.cash += self.inventory * mid_next
            self.inventory = 0
        return StepResult(
            state=self.data.states[self.t],
            aux=self._aux(),
            reward=float(reward),
            done=done,
            info=info,
        )
