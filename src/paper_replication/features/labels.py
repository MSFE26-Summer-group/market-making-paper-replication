"""Mid-price direction labels (paper Eq. 5-7).

We follow the DeepLOB formulation the paper cites as its label source ([23]
Zhang & Zohren): compare a trailing k-point mid-price average against a
leading k-point average, rather than the paper's own ambiguous ``t-/+i``
notation, since the paper explicitly borrows the labeling approach from [23].
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _trailing_leading_means(
    mid_price: FloatArray, k: int
) -> tuple[FloatArray, FloatArray]:
    """past_mean(t) = mean(p[t-k+1..t]); future_mean(t) = mean(p[t+1..t+k]).

    NaN outside the valid range [k-1, n-1-k].
    """
    n = len(mid_price)
    csum: FloatArray = np.concatenate(([0.0], np.cumsum(mid_price)))

    past_mean = np.full(n, np.nan)
    valid_past = np.arange(k - 1, n)
    past_mean[valid_past] = (csum[valid_past + 1] - csum[valid_past + 1 - k]) / k

    future_mean = np.full(n, np.nan)
    valid_future = np.arange(0, n - k)
    future_mean[valid_future] = (
        csum[valid_future + 1 + k] - csum[valid_future + 1]
    ) / k

    return past_mean, future_mean


def compute_l_t(mid_price: FloatArray, k: int) -> FloatArray:
    """Relative change between future and past k-point mid-price averages (Eq. 6-7)."""
    past_mean, future_mean = _trailing_leading_means(mid_price, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        l_t: FloatArray = (future_mean - past_mean) / past_mean
    return l_t


def calibrate_alpha(l_t: FloatArray, target_balance: float = 1.0 / 3.0) -> float:
    """Pick alpha so ~target_balance of samples land in each of the up/down classes.

    The paper's alpha=1e-5 was calibrated to a different asset's tick-level
    noise and doesn't transfer numerically to our ~10s BTC/USDT snapshots, so
    we calibrate from the empirical distribution of `l_t` instead.
    """
    if not 0 < target_balance < 0.5:
        raise ValueError("target_balance must be in (0, 0.5)")
    finite = l_t[np.isfinite(l_t)]
    if finite.size == 0:
        raise ValueError("l_t has no finite values to calibrate against")
    return float(np.quantile(np.abs(finite), 1 - 2 * target_balance))


def direction_labels(l_t: FloatArray, alpha: float) -> FloatArray:
    """Eq. 5: +1 up, -1 down, 0 stationary. NaN propagates where `l_t` is NaN."""
    labels: FloatArray = np.full(l_t.shape, np.nan)
    finite = np.isfinite(l_t)
    labels[finite & (l_t > alpha)] = 1.0
    labels[finite & (l_t < -alpha)] = -1.0
    labels[finite & (l_t >= -alpha) & (l_t <= alpha)] = 0.0
    return labels
