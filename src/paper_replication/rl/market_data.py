"""Assembles RL environment inputs from a LOB snapshot file (paper III-A, IV-C1).

Reuses `features.lob_state` and `features.dynamic_state` -- the same
window/normalization/dynamic-feature machinery the Attn-LOB pretraining
pipeline uses -- but skips `features.labels` entirely: the RL agent doesn't
need a precomputed direction label, it needs the raw best bid/ask/mid price
at each step to execute simulated quotes against (see `simulator.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from paper_replication.features.config import FeatureConfig
from paper_replication.features.dynamic_state import (
    order_strength_index_from_ticks,
    order_strength_index_proxy,
    realized_volatility,
    relative_strength_index,
)
from paper_replication.features.lob_state import (
    level_column_names,
    load_lob_snapshot,
    lob_state_matrix,
    normalize_lob_state,
    rolling_windows,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass
class MarketDataset:
    """Aligned per-step market data ready to drive `MarketMakingEnv`.

    lob_state: (n_steps, window_T, 4*n_levels), normalized -- Attn-LOB input.
    dynamic_state: (n_steps, n_dynamic_features).
    best_ask / best_bid / mid_price: (n_steps,) raw level-1 prices used by
        `simulator.MarketSimulator` to fill agent quotes -- NOT normalized.
    timestamps: (n_steps,) snapshot timestamp of each step (window's last row).
    """

    lob_state: FloatArray
    dynamic_state: FloatArray
    dynamic_feature_names: list[str]
    best_ask: FloatArray
    best_bid: FloatArray
    mid_price: FloatArray
    timestamps: FloatArray

    def __len__(self) -> int:
        return len(self.timestamps)


def build_market_dataset(
    lob_parquet_path: str,
    config: FeatureConfig,
    symbol: str = "BTCUSDT",
    ticks_parquet_path: str | None = None,
) -> MarketDataset:
    """Loads the LOB snapshot file and builds an unlabeled RL market dataset.

    Mirrors `features.dataset.build_feature_dataset`'s window/dynamic-state
    construction and row alignment, minus the label machinery: the RL
    environment computes its own reward from `best_ask`/`best_bid`/`mid_price`
    rather than a precomputed direction label, so there's no `horizon_k`
    look-ahead trim -- every row from the first full window onward is usable.
    """
    df = load_lob_snapshot(lob_parquet_path, symbol=symbol, n_levels=config.n_levels)
    n = len(df)

    raw_state = lob_state_matrix(df, symbol, config.n_levels)
    windows = rolling_windows(raw_state, config.window_T)
    lob_state_norm = normalize_lob_state(windows, config.n_levels)

    rv = realized_volatility(df, config.rv_windows_seconds)
    rsi = relative_strength_index(df, config.rsi_windows_seconds)
    if config.use_real_osi:
        if ticks_parquet_path is None:
            raise ValueError("config.use_real_osi=True requires ticks_parquet_path")
        osi = order_strength_index_from_ticks(
            df, ticks_parquet_path, config.osi_windows_seconds
        )
    else:
        osi = order_strength_index_proxy(df, config.osi_windows_seconds)
    dynamic = pd.concat([rv, rsi, osi], axis=1)

    level_cols = level_column_names(symbol, config.n_levels)
    best_ask_col, best_bid_col = level_cols[0], level_cols[2]
    best_ask = df[best_ask_col].to_numpy(dtype=np.float64)
    best_bid = df[best_bid_col].to_numpy(dtype=np.float64)
    mid_price = df["mid_price"].to_numpy(dtype=np.float64)

    # Window i's last row is source row (window_T - 1 + i) -- same offset
    # `features.dataset.build_feature_dataset` uses to line windows back up
    # with per-row data (lob_state.rolling_windows's docstring).
    window_start = config.window_T - 1
    rows = np.arange(window_start, n)
    window_idx = rows - window_start

    lob_state_sel = lob_state_norm[window_idx]
    dynamic_sel = dynamic.to_numpy()[rows]
    best_ask_sel = best_ask[rows]
    best_bid_sel = best_bid[rows]
    mid_price_sel = mid_price[rows]
    timestamps_sel = df["timestamp"].to_numpy()[rows]

    valid = (
        np.isfinite(dynamic_sel).all(axis=1)
        & np.isfinite(best_ask_sel)
        & np.isfinite(best_bid_sel)
        & np.isfinite(mid_price_sel)
    )
    if not valid.all():
        lob_state_sel = lob_state_sel[valid]
        dynamic_sel = dynamic_sel[valid]
        best_ask_sel = best_ask_sel[valid]
        best_bid_sel = best_bid_sel[valid]
        mid_price_sel = mid_price_sel[valid]
        timestamps_sel = timestamps_sel[valid]

    return MarketDataset(
        lob_state=lob_state_sel,
        dynamic_state=dynamic_sel,
        dynamic_feature_names=list(dynamic.columns),
        best_ask=best_ask_sel,
        best_bid=best_bid_sel,
        mid_price=mid_price_sel,
        timestamps=timestamps_sel,
    )


def chronological_split(
    dataset: MarketDataset, test_frac: float = 0.3
) -> dict[str, MarketDataset]:
    """Time-ordered train/test split -- no shuffling, so no lookahead leakage.

    Simpler than `features.dataset.chronological_split`'s train/val/test:
    RL evaluation compares trained policies and baselines on the same
    held-out test episodes, so there's no separate model-selection split to
    carve out. `MarketMakingEnv` slices each half into its own
    non-overlapping episodes independently, so the split point itself can't
    leak information across an episode boundary.
    """
    n = len(dataset)
    split = int(round(n * (1 - test_frac)))

    def _slice(lo: int, hi: int) -> MarketDataset:
        return MarketDataset(
            lob_state=dataset.lob_state[lo:hi],
            dynamic_state=dataset.dynamic_state[lo:hi],
            dynamic_feature_names=dataset.dynamic_feature_names,
            best_ask=dataset.best_ask[lo:hi],
            best_bid=dataset.best_bid[lo:hi],
            mid_price=dataset.mid_price[lo:hi],
            timestamps=dataset.timestamps[lo:hi],
        )

    return {"train": _slice(0, split), "test": _slice(split, n)}
