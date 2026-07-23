"""Orchestrates building the labeled Attn-LOB pretraining dataset."""

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
from paper_replication.features.labels import (
    calibrate_alpha,
    compute_l_t,
    direction_labels,
)
from paper_replication.features.lob_state import (
    load_lob_snapshot,
    lob_state_matrix,
    normalize_lob_state,
    rolling_windows,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int8]


@dataclass
class AttnLOBDataset:
    """Aligned, labeled samples ready to feed the Attn-LOB model.

    lob_state: (n_samples, window_T, 4 * n_levels), z-norm/max-norm applied.
    dynamic_state: (n_samples, n_dynamic_features).
    labels: (n_samples,) int8 in {-1, 0, 1}.
    timestamps: (n_samples,) snapshot timestamp of each window's last row.
    """

    lob_state: FloatArray
    dynamic_state: FloatArray
    dynamic_feature_names: list[str]
    labels: IntArray
    timestamps: FloatArray
    alpha_used: float

    def __len__(self) -> int:
        return len(self.labels)


def build_feature_dataset(
    lob_parquet_path: str,
    config: FeatureConfig,
    symbol: str = "BTCUSDT",
    ticks_parquet_path: str | None = None,
) -> AttnLOBDataset:
    """Load the LOB snapshot file and build a labeled Attn-LOB dataset from it.

    By default OSI is proxied from columns already in `lob_parquet_path`
    (see `dynamic_state.order_strength_index_proxy`). Pass
    `config.use_real_osi=True` and a `ticks_parquet_path` to instead recompute
    it from raw trade ticks -- much slower, not needed for a first pass.
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

    l_t = compute_l_t(df["mid_price"].to_numpy(), config.horizon_k)
    alpha = (
        config.label_alpha
        if config.label_alpha is not None
        else calibrate_alpha(l_t, config.label_target_balance)
    )
    labels = direction_labels(l_t, alpha)

    # Window i's last row is source row (window_T - 1 + i); labels/dynamic
    # state are computed per source row, so align everything on row index.
    window_start = config.window_T - 1
    label_lo, label_hi = config.horizon_k - 1, n - config.horizon_k
    lo = max(window_start, label_lo)
    hi = min(n - 1, label_hi)
    if lo > hi:
        raise ValueError("No valid rows: window_T/horizon_k too large for this dataset")

    rows = np.arange(lo, hi + 1)
    window_idx = rows - window_start

    lob_state_sel = lob_state_norm[window_idx]
    dynamic_sel = dynamic.to_numpy()[rows]
    labels_sel = labels[rows]
    timestamps_sel = df["timestamp"].to_numpy()[rows]

    valid = np.isfinite(labels_sel) & np.isfinite(dynamic_sel).all(axis=1)
    if not valid.all():
        lob_state_sel = lob_state_sel[valid]
        dynamic_sel = dynamic_sel[valid]
        labels_sel = labels_sel[valid]
        timestamps_sel = timestamps_sel[valid]

    return AttnLOBDataset(
        lob_state=lob_state_sel,
        dynamic_state=dynamic_sel,
        dynamic_feature_names=list(dynamic.columns),
        labels=labels_sel.astype(np.int8),
        timestamps=timestamps_sel,
        alpha_used=alpha,
    )


def chronological_split(
    dataset: AttnLOBDataset, test_frac: float = 0.5, val_frac_of_train: float = 0.2
) -> dict[str, AttnLOBDataset]:
    """Time-ordered train/val/test split -- no shuffling, so no lookahead leakage.

    Mirrors the paper's split (first half of the month for train+val, second
    half for test), adapted to however much history `dataset` covers.
    """
    n = len(dataset)
    test_start = int(round(n * (1 - test_frac)))
    val_start = int(round(test_start * (1 - val_frac_of_train)))

    def _slice(lo: int, hi: int) -> AttnLOBDataset:
        return AttnLOBDataset(
            lob_state=dataset.lob_state[lo:hi],
            dynamic_state=dataset.dynamic_state[lo:hi],
            dynamic_feature_names=dataset.dynamic_feature_names,
            labels=dataset.labels[lo:hi],
            timestamps=dataset.timestamps[lo:hi],
            alpha_used=dataset.alpha_used,
        )

    return {
        "train": _slice(0, val_start),
        "val": _slice(val_start, test_start),
        "test": _slice(test_start, n),
    }
