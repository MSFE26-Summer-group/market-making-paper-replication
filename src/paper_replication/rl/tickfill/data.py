"""LOB data pipeline.

Loads 10-second BTC-USDT LOB snapshots and produces normalized
state tensors for the RL agent, following Guo, Lin & Huang (2023):
10 levels x (bid price, bid size, ask price, ask size) = 40 features
per snapshot, stacked over a lookback window.

Adaptations vs the paper (documented for the replication report):
- Prices are expressed as distance to mid in basis points instead of
  raw z-scores: BTC trends over the 11-day sample, so raw-price
  z-scores are non-stationary. Relative prices are what a market
  maker actually quotes against.
- Sizes are log1p-transformed before z-scoring (heavy right tail).
- Normalization statistics are estimated on the training split only.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_LEVELS = 10
SYMBOL = "BTCUSDT"

LOB_COLS = [
    f"{SYMBOL}.{side}_{kind}_{lvl}"
    for lvl in range(1, N_LEVELS + 1)
    for side in ("bid", "ask")
    for kind in ("price", "size")
]
PRICE_COLS = [c for c in LOB_COLS if "price" in c]
SIZE_COLS = [c for c in LOB_COLS if "size" in c]

AUX_COLS = [
    "mid_price",
    "max_trade_price",
    "min_trade_price",
    "lob_imbalance",
    "best_bid",
    "best_ask",
]


@dataclass
class LOBDataset:
    """Normalized LOB states plus the raw series the environment needs."""

    states: np.ndarray  # (N, window, 40) normalized model input
    mid: np.ndarray  # (N,) raw mid price
    trade_max: np.ndarray  # (N,) highest trade price in the 10s interval
    trade_min: np.ndarray  # (N,) lowest trade price in the interval
    imbalance: np.ndarray  # (N,) LOB imbalance, the paper's OSI analogue
    spread: np.ndarray  # (N,) best ask - best bid, for ND-PnL metric
    size_mean: np.ndarray  # (40-20,) train-split stats, kept for reuse
    size_std: np.ndarray
    # Side-aware fill prices derived from the tick tape (None -> the env
    # falls back to the coarser all-trades min/max above).
    sell_min: np.ndarray | None = None  # min price of seller-initiated trades
    buy_max: np.ndarray | None = None  # max price of buyer-initiated trades


def load_lob(path: str) -> pd.DataFrame:
    """Read the snapshot parquet and keep only the columns we use."""
    df = pd.read_parquet(path, columns=LOB_COLS + AUX_COLS + ["timestamp"])
    df = df.dropna(subset=LOB_COLS + ["mid_price"]).reset_index(drop=True)
    return df


def build_dataset(
    df: pd.DataFrame, window: int = 50, train_frac: float = 0.7
) -> LOBDataset:
    """Turn raw snapshots into stacked, normalized state windows.

    Prices -> basis-point distance to the same row's mid price.
    Sizes  -> log1p, then z-score with stats from the first
    ``train_frac`` of rows only (no look-ahead).
    """
    mid = df["mid_price"].to_numpy(dtype=np.float64)

    prices = df[PRICE_COLS].to_numpy(dtype=np.float64)
    rel_prices = (prices / mid[:, None] - 1.0) * 1e4  # basis points

    sizes = np.log1p(df[SIZE_COLS].to_numpy(dtype=np.float64))
    n_train = int(len(df) * train_frac)
    size_mean = sizes[:n_train].mean(axis=0)
    size_std = sizes[:n_train].std(axis=0) + 1e-8
    z_sizes = (sizes - size_mean) / size_std

    feats = np.empty((len(df), len(LOB_COLS)), dtype=np.float32)
    feats[:, 0::2] = rel_prices.astype(np.float32)
    feats[:, 1::2] = z_sizes.astype(np.float32)

    # windows[i] = feats[i-window+1 .. i], aligned so states[i] uses only
    # information available at snapshot i.
    idx = np.arange(window)[None, :] + np.arange(len(df) - window + 1)[:, None]
    states = feats[idx]

    off = window - 1  # first row with a full lookback window
    best_bid = df["best_bid"].to_numpy(dtype=np.float64)
    best_ask = df["best_ask"].to_numpy(dtype=np.float64)
    ds = LOBDataset(
        states=states,
        mid=mid[off:],
        trade_max=df["max_trade_price"].to_numpy(dtype=np.float64)[off:],
        trade_min=df["min_trade_price"].to_numpy(dtype=np.float64)[off:],
        imbalance=df["lob_imbalance"].to_numpy(dtype=np.float64)[off:],
        spread=(best_ask - best_bid)[off:],
        size_mean=size_mean,
        size_std=size_std,
    )
    if "tick_sell_min" in df.columns:
        ds.sell_min = df["tick_sell_min"].to_numpy(dtype=np.float64)[off:]
        ds.buy_max = df["tick_buy_max"].to_numpy(dtype=np.float64)[off:]
    return ds


def attach_tick_fills(df: pd.DataFrame, ticks_path: str) -> pd.DataFrame:
    """Add side-aware fill columns to the snapshot frame from the tick tape.

    For each snapshot interval (t_{i-1}, t_i] we record:
    - tick_sell_min: lowest price among seller-initiated trades (side=-1).
      A resting bid can only be hit by sellers, so this is the price our
      bid must beat to fill.
    - tick_buy_max: highest price among buyer-initiated trades (side=+1),
      the analogue for our resting ask.

    NaN means no trades of that side in the interval (quote cannot fill).
    """
    ticks = pd.read_parquet(ticks_path, columns=["timestamp", "price", "side"])
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    # Assign each tick to the snapshot row whose interval contains it.
    idx = np.searchsorted(ts, ticks["timestamp"].to_numpy(), side="left")
    valid = (idx > 0) & (idx < len(ts))
    ticks = ticks.iloc[valid].assign(row=idx[valid])

    sell = ticks[ticks["side"] == -1].groupby("row")["price"].min()
    buy = ticks[ticks["side"] == 1].groupby("row")["price"].max()
    out = df.copy()
    out["tick_sell_min"] = out.index.map(sell)
    out["tick_buy_max"] = out.index.map(buy)
    return out
