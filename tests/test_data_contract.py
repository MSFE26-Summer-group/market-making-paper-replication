"""Data-contract checks: verify derived columns against raw sources.

Lesson from Run 1: the snapshot's trade-stat columns were assumed to be
trailing-interval trade ranges, but actually cover ~1s after each label.
Any column used as a fill referee must be validated against the tick
tape BEFORE use. These tests encode that contract; they are skipped
when the data files are absent (CI has no data).
"""

from pathlib import Path

import numpy as np
import pytest

LOB = Path("data/btc_usdt_20221019_20221030_lob.parquet")
TICKS = Path("data/btc_usdt_20221019_20221030_ticks.parquet")

pytestmark = pytest.mark.skipif(
    not (LOB.exists() and TICKS.exists()), reason="data files not present"
)


@pytest.fixture(scope="module")
def frames():
    import pandas as pd

    lob = pd.read_parquet(
        LOB, columns=["timestamp", "mid_price", "min_trade_price", "max_trade_price"]
    )
    ticks = pd.read_parquet(TICKS, columns=["timestamp", "price"]).sort_values(
        "timestamp"
    )
    return lob, ticks


def interval_stats(lob, ticks):
    ts = lob["timestamp"].to_numpy()
    idx = np.searchsorted(ts, ticks["timestamp"].to_numpy(), side="left")
    valid = (idx > 0) & (idx < len(ts))
    t = ticks.iloc[valid].assign(row=idx[valid])
    g = t.groupby("row")["price"]
    return lob.index.map(g.min()), lob.index.map(g.max())


class TestTickTapeIsTheFillReferee:
    def test_tape_aligns_with_book(self, frames):
        """Last trade of each interval must sit near the row's mid."""
        import pandas as pd

        lob, ticks = frames
        ts = lob["timestamp"].to_numpy()
        idx = np.searchsorted(ts, ticks["timestamp"].to_numpy(), side="left")
        valid = (idx > 0) & (idx < len(ts))
        t = ticks.iloc[valid].assign(row=idx[valid])
        last = lob.index.map(t.groupby("row")["price"].last())
        bps = (
            np.abs(last - lob["mid_price"]) / lob["mid_price"] * 1e4
        )
        assert np.nanmedian(bps) < 1.0  # tape and book on the same market/clock

    def test_trade_stat_columns_are_not_interval_ranges(self, frames):
        """Documents the Run 1 root cause: the precomputed columns do NOT
        match the interval's true trade range — anyone tempted to use
        them as a fill referee should see this test."""
        lob, ticks = frames
        tmin, _ = interval_stats(lob, ticks)
        m = np.isfinite(tmin)
        match = np.isclose(
            lob["min_trade_price"].to_numpy()[m], np.asarray(tmin)[m], atol=0.01
        ).mean()
        assert match < 0.5  # if this ever PASSES 50%, the data changed - re-audit
