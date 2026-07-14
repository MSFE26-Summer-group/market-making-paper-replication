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

AUX_COLS = ["mid_price", "max_trade_price", "min_trade_price", "lob_imbalance"]


@dataclass
class LOBDataset:
    """Normalized LOB states plus the raw series the environment needs."""

    states: np.ndarray  # (N, window, 40) normalized model input
    mid: np.ndarray  # (N,) raw mid price
    trade_max: np.ndarray  # (N,) highest trade price in the 10s interval
    trade_min: np.ndarray  # (N,) lowest trade price in the interval
    imbalance: np.ndarray  # (N,) LOB imbalance, the paper's OSI analogue
    size_mean: np.ndarray  # (40-20,) train-split stats, kept for reuse
    size_std: np.ndarray


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
    return LOBDataset(
        states=states,
        mid=mid[off:],
        trade_max=df["max_trade_price"].to_numpy(dtype=np.float64)[off:],
        trade_min=df["min_trade_price"].to_numpy(dtype=np.float64)[off:],
        imbalance=df["lob_imbalance"].to_numpy(dtype=np.float64)[off:],
        size_mean=size_mean,
        size_std=size_std,
    )
