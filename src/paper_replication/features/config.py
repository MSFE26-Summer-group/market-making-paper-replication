"""Configuration for the LOB feature-engineering pipeline (Attn-LOB pretraining)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    """Hyperparameters for building the Attn-LOB pretraining dataset.

    The source paper defines `window_T` and `horizon_k` in units of raw LOB
    *events* (~60-150ms apart). Our data is a fixed ~10s snapshot grid, so
    the same row-count parameters correspond to a much longer wall-clock
    span (T=50 rows ~= 8 minutes of context, k=10 rows ~= 100s label
    horizon). `osi_windows_seconds` / `rv_windows_seconds` / `rsi_windows_seconds`
    stay genuinely time-based (as in the paper) since those are defined in
    real seconds, not event counts.
    """

    n_levels: int = 10
    window_T: int = 50
    horizon_k: int = 10

    # None => calibrate from data so classes are ~label_target_balance each.
    label_alpha: float | None = None
    label_target_balance: float = 1.0 / 3.0

    osi_windows_seconds: tuple[int, ...] = (10, 60, 300)
    rv_windows_seconds: tuple[int, ...] = (300, 600, 1800)
    rsi_windows_seconds: tuple[int, ...] = (300, 600, 1800)

    # False (default): OSI is proxied from columns already in the LOB
    # snapshot file. True: recompute OSI from raw trade ticks (requires
    # ticks_parquet_path and is much more expensive -- see dynamic_state.py).
    use_real_osi: bool = False
