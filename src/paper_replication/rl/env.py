"""Market-making environment over historical LOB snapshots.

Gym-style episodic environment following Guo, Lin & Huang (2023):

- Action: continuous (reservation-price bias, half-spread), both in
  basis points of the current mid price.
- Fill model: the agent's quotes rest for one 10s interval. With tick
  data attached (LOBDataset.sell_min/buy_max) fills are side-aware: a
  bid fills only if a seller-initiated trade printed at or below it,
  an ask only if a buyer-initiated trade printed at or above it.
  Without tick data it falls back to the coarser all-trades range.
  One unit per fill, no market impact (historical replay).
- Reward: trading PnL + dampened holding PnL - inventory penalty,
  with the paper's eta = 0.5 dampening.
- Agent state (aux) follows the paper: inventory, order imbalance
  (OSI analogue) and elapsed episode time fraction.
- Inventory is liquidated at mid at the end of each episode.
- ``start_range`` restricts episode starts, giving disjoint train and
  test segments on the same dataset.
"""

from dataclasses import dataclass, field

import numpy as np

from paper_replication.rl.data import LOBDataset


@dataclass
class EnvConfig:
    episode_len: int = 360  # 360 x 10s = 1 hour
    # Paper (A-shares): max_bias=0.05 CNY, max_spread=0.1 CNY on ~15-50
    # CNY stocks, i.e. roughly 10-30 bps. BTC-USDT quotes ~0.5-1 bp wide,
    # so we keep the caps in bps and of comparable magnitude.
    max_bias_bps: float = 5.0  # |reservation bias| cap
    min_half_spread_bps: float = 0.5
    max_half_spread_bps: float = 10.0
    dampen: float = 0.5  # eta in the paper's dampened PnL (eq. 13)
    inv_penalty: float = 0.01  # zeta * q^2 (eq. 15), in bps-of-mid units
    max_inventory: int = 10  # omega * minimum_trade_unit position cap


@dataclass
class StepResult:
    state: np.ndarray
    aux: np.ndarray  # (inventory/max_inv, lob_imbalance, time_frac)
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
        start_range: tuple[int, int] | None = None,
    ) -> None:
        self.data = data
        self.cfg = config or EnvConfig()
        self.rng = rng or np.random.default_rng(0)
        lo, hi = start_range or (0, len(data.mid))
        self.start_lo = lo
        self.start_hi = min(hi, len(data.mid)) - self.cfg.episode_len - 1
        if self.start_hi <= self.start_lo:
            raise ValueError("start_range too small for episode_len")
        self.t = 0
        self.start = 0
        self.inventory = 0
        self.cash = 0.0

    def _aux(self) -> np.ndarray:
        return np.array(
            [
                self.inventory / self.cfg.max_inventory,
                self.data.imbalance[self.t],
                (self.t - self.start) / self.cfg.episode_len,
            ],
            dtype=np.float32,
        )

    def reset(self, start: int | None = None) -> StepResult:
        """Start a new episode at ``start`` (random in range if omitted)."""
        self.start = (
            int(self.rng.integers(self.start_lo, self.start_hi))
            if start is None
            else start
        )
        self.t = self.start
        self.inventory = 0
        self.cash = 0.0
        return StepResult(
            state=self.data.states[self.t], aux=self._aux(), reward=0.0, done=False
        )

    def _fills(self, nxt: int, bid: float, ask: float) -> tuple[bool, bool]:
        """Side-aware fill check when tick data is attached, else coarse."""
        if self.data.sell_min is not None and self.data.buy_max is not None:
            sell_min = self.data.sell_min[nxt]
            buy_max = self.data.buy_max[nxt]
            buy_fill = bool(sell_min == sell_min and sell_min <= bid)
            sell_fill = bool(buy_max == buy_max and buy_max >= ask)
        else:
            buy_fill = bool(self.data.trade_min[nxt] <= bid)
            sell_fill = bool(self.data.trade_max[nxt] >= ask)
        return buy_fill, sell_fill

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

        raw_buy, raw_sell = self._fills(nxt, bid, ask)
        buy_fill = raw_buy and self.inventory < cfg.max_inventory
        sell_fill = raw_sell and self.inventory > -cfg.max_inventory
        if buy_fill:
            self.inventory += 1
            self.cash -= bid
            trading_pnl += mid_next - bid
        if sell_fill:
            self.inventory -= 1
            self.cash += ask
            trading_pnl += ask - mid_next

        # Holding PnL on the position carried into the interval, with the
        # speculative (positive) part dampened (eta).
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
            "fills": float(buy_fill) + float(sell_fill),
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
