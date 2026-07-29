"""Tests for paper_replication.features.dynamic_state.

The rolling windows used here are deliberately much longer than the total
span of the synthetic series, so every point falls inside every window
regardless of pandas' open/closed time-window boundary convention -- that
lets us hand-verify results as plain cumulative sums/means.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_replication.features.dynamic_state import (
    order_strength_index_from_ticks,
    order_strength_index_proxy,
    realized_volatility,
    relative_strength_index,
)

HUGE_WINDOW = (1_000_000,)


@pytest.fixture
def snapshot_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [0.0, 10.0, 20.0, 30.0, 40.0],
            "mid_price": [100.0, 101.0, 99.0, 99.0, 105.0],
            "net_trade_sign": [2, -1, 3, 0, 1],
            "trade_count": [5, 4, 6, 0, 2],
            "ema_ofi": [0.1, -0.2, 0.3, 0.0, 0.05],
        }
    )


def test_realized_volatility_full_window_matches_manual_sum(
    snapshot_df: pd.DataFrame,
) -> None:
    log_ret = np.diff(np.log(snapshot_df["mid_price"].to_numpy()))
    expected_last = np.sqrt(np.sum(log_ret**2))

    rv = realized_volatility(snapshot_df, HUGE_WINDOW)

    np.testing.assert_allclose(rv.iloc[-1, 0], expected_last)
    assert np.isnan(rv.iloc[0, 0])  # no prior return at the first point


def test_relative_strength_index_full_window_matches_manual_ratio(
    snapshot_df: pd.DataFrame,
) -> None:
    diffs = np.diff(snapshot_df["mid_price"].to_numpy())
    gains = np.clip(diffs, 0, None).sum()
    losses = np.clip(-diffs, 0, None).sum()
    expected_last = gains / (gains + losses)

    rsi = relative_strength_index(snapshot_df, HUGE_WINDOW)

    np.testing.assert_allclose(rsi.iloc[-1, 0], expected_last)


def test_order_strength_index_proxy_full_window_matches_manual_aggregates(
    snapshot_df: pd.DataFrame,
) -> None:
    expected_osi_n = (
        snapshot_df["net_trade_sign"].sum() / snapshot_df["trade_count"].sum()
    )
    expected_osi_v = snapshot_df["ema_ofi"].mean()

    osi = order_strength_index_proxy(snapshot_df, HUGE_WINDOW)

    np.testing.assert_allclose(osi["osi_n_1000000s"].iloc[-1], expected_osi_n)
    np.testing.assert_allclose(osi["osi_v_1000000s"].iloc[-1], expected_osi_v)


def test_order_strength_index_from_ticks_matches_manual_aggregates(
    snapshot_df: pd.DataFrame, tmp_path: Path
) -> None:
    ticks = pd.DataFrame(
        {
            "timestamp": [1.0, 5.0, 15.0, 25.0, 35.0],
            "side": [1, -1, 1, 1, -1],
            "amount": [0.5, 0.2, 0.3, 0.1, 0.4],
            "price": [100.0, 100.5, 99.5, 99.0, 104.0],
        }
    )
    ticks_path = tmp_path / "ticks.parquet"
    ticks.to_parquet(ticks_path)

    osi = order_strength_index_from_ticks(snapshot_df, str(ticks_path), HUGE_WINDOW)

    buy = ticks[ticks["side"] == 1]
    sell = ticks[ticks["side"] == -1]
    expected_osi_v = (buy["amount"].sum() - sell["amount"].sum()) / ticks[
        "amount"
    ].sum()
    expected_osi_n = (len(buy) - len(sell)) / len(ticks)

    np.testing.assert_allclose(osi["osi_v_1000000s"].iloc[-1], expected_osi_v)
    np.testing.assert_allclose(osi["osi_n_1000000s"].iloc[-1], expected_osi_n)
