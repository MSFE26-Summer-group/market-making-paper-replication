"""Raw LOB state: snapshot loading, rolling windows, normalization (paper III-A1)."""

from __future__ import annotations

from typing import TypeAlias

import duckdb
import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray: TypeAlias = npt.NDArray[np.float64]

# Extra (non-level) columns pulled from the snapshot file, needed by the
# dynamic-state features in dynamic_state.py.
_EXTRA_COLUMNS = ["mid_price", "net_trade_sign", "trade_count", "ema_ofi"]


def level_column_names(symbol: str, n_levels: int) -> list[str]:
    """Column names for `n_levels` of ask/bid price/size.

    Ordered [ask_price_i, ask_size_i, bid_price_i, bid_size_i] per level i,
    matching the paper's LOB state vector (Eq. 1).
    """
    names: list[str] = []
    for i in range(1, n_levels + 1):
        names += [
            f"{symbol}.ask_price_{i}",
            f"{symbol}.ask_size_{i}",
            f"{symbol}.bid_price_{i}",
            f"{symbol}.bid_size_{i}",
        ]
    return names


def load_lob_snapshot(parquet_path: str, symbol: str, n_levels: int) -> pd.DataFrame:
    """Load timestamp, level columns, and the extra columns dynamic_state.py needs.

    Rows are sorted by timestamp and de-duplicated on timestamp.
    """
    level_cols = level_column_names(symbol, n_levels)
    select_cols = ", ".join(f'"{c}"' for c in [*level_cols, *_EXTRA_COLUMNS])
    query = f"""
        SELECT timestamp, {select_cols}
        FROM '{parquet_path}'
        WHERE mid_price IS NOT NULL
        ORDER BY timestamp
    """
    df = duckdb.query(query).df()
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df


def lob_state_matrix(df: pd.DataFrame, symbol: str, n_levels: int) -> FloatArray:
    """The raw (n_rows, 4 * n_levels) LOB state matrix, per paper Eq. 1."""
    cols = level_column_names(symbol, n_levels)
    matrix: FloatArray = df[cols].to_numpy(dtype=np.float64)
    return matrix


def rolling_windows(matrix: FloatArray, window_T: int) -> FloatArray:
    """Overlapping windows of length `window_T`.

    (n_rows, F) -> (n_rows - window_T + 1, window_T, F). Window i's last row
    is source row `i + window_T - 1`.
    """
    if len(matrix) < window_T:
        raise ValueError(f"Need at least {window_T} rows, got {len(matrix)}")
    view = np.lib.stride_tricks.sliding_window_view(
        matrix, window_shape=window_T, axis=0
    )
    # sliding_window_view appends the window axis last -> (n_windows, F, window_T).
    result: FloatArray = np.ascontiguousarray(np.moveaxis(view, -1, 1))
    return result


def normalize_lob_state(windows: FloatArray, n_levels: int) -> FloatArray:
    """Per-window z-norm on price columns, max-norm on volume columns (paper III-B2)."""
    n_features = windows.shape[-1]
    expected = 4 * n_levels
    if n_features != expected:
        raise ValueError(
            f"Expected {expected} features (4 * n_levels), got {n_features}"
        )

    price_mask = np.tile(np.array([True, False, True, False]), n_levels)
    vol_mask = ~price_mask

    out = windows.astype(np.float64, copy=True)

    price = out[..., price_mask]
    mean = price.mean(axis=1, keepdims=True)
    std = price.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    out[..., price_mask] = (price - mean) / std

    vol = out[..., vol_mask]
    vol_max = vol.max(axis=1, keepdims=True)
    vol_max = np.where(vol_max == 0, 1.0, vol_max)
    out[..., vol_mask] = vol / vol_max

    result: FloatArray = out
    return result
