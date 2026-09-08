"""Build tick-fill inputs for the monthly (2022-10-10 .. 2022-11-10) LOB file.

The monthly snapshot file lacks mid_price and the tick tape only covers
2022-10-19 .. 2022-10-30, so this script:

1. Writes a derived monthly LOB parquet with mid_price (+ zero-filled
   placeholder columns the shared loader selects but Exp 4 never reads).
2. Downloads Binance daily aggTrades for the days the tick file misses,
   reduces every day (existing tape + downloads) to per-snapshot-interval
   extremes: min seller-initiated print and max buyer-initiated print.
3. Validates the two sources against each other on an overlap day
   (2022-10-25) before trusting the downloads.
4. Emits a tiny extremes parquet with (timestamp, price, side) rows that
   reproduce identical fills through build_exp4_data's ticks_parquet path.

    PYTHONPATH=src .venv/bin/python scripts/build_monthly_tick_extremes.py
"""

from __future__ import annotations

import io
import sys
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

LOB_IN = "data/btc_usdt_20221010_20221110_lob.parquet"
LOB_OUT = "data/btc_usdt_20221010_20221110_lob_derived.parquet"
TICKS_IN = "data/btc_usdt_20221019_20221030_ticks.parquet"
EXTREMES_OUT = "data/btc_usdt_20221010_20221110_tick_extremes.parquet"
CACHE = Path("data/aggtrades_cache")

DAY0, DAY1 = date(2022, 10, 10), date(2022, 11, 10)
TAPE0, TAPE1 = date(2022, 10, 19), date(2022, 10, 30)  # existing tape coverage
OVERLAP_CHECK = date(2022, 10, 25)


def day_range(a: date, b: date) -> list[date]:
    return [a + timedelta(days=i) for i in range((b - a).days + 1)]


def fetch_aggtrades(day: date) -> pd.DataFrame:
    """One day of Binance spot aggTrades as (timestamp[s], price, side)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    zpath = CACHE / f"BTCUSDT-aggTrades-{day}.zip"
    if not zpath.exists():
        url = (
            "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/"
            f"BTCUSDT-aggTrades-{day}.zip"
        )
        t0 = time.time()
        with urlopen(url) as r:
            zpath.write_bytes(r.read())
        print(f"  downloaded {zpath.name} in {time.time()-t0:.0f}s", flush=True)
    with zipfile.ZipFile(zpath) as z:
        raw = z.read(z.namelist()[0])
    first = raw[:64].split(b"\n", 1)[0]
    header = 0 if b"agg" in first.lower() else None
    df = pd.read_csv(
        io.BytesIO(raw),
        header=header,
        usecols=[1, 5, 6],
        names=None if header == 0 else ["p", "t", "m"],
    )
    df.columns = ["p", "t", "m"]
    # isBuyerMaker=True -> seller initiated -> side -1 (matches tape convention)
    return pd.DataFrame(
        {
            "timestamp": df["t"].to_numpy(np.float64) / 1000.0,
            "price": df["p"].to_numpy(np.float64),
            "side": np.where(df["m"].to_numpy(bool), -1, 1).astype(np.int32),
        }
    )


def accumulate(
    ts: np.ndarray,
    price: np.ndarray,
    side: np.ndarray,
    grid: np.ndarray,
    sell_min: np.ndarray,
    buy_max: np.ndarray,
) -> None:
    """Fold one batch of prints into per-interval extremes (same searchsorted
    convention as build_exp4_data)."""
    idx = np.searchsorted(grid, ts, side="left")
    ok = (idx > 0) & (idx < len(grid))
    idx, price, side = idx[ok], price[ok], side[ok]
    for s, arr, red in ((-1, sell_min, np.fmin), (1, buy_max, np.fmax)):
        m = side == s
        if m.any():
            grp = pd.Series(price[m]).groupby(idx[m])
            agg = grp.min() if s == -1 else grp.max()
            arr[agg.index.to_numpy()] = red(arr[agg.index.to_numpy()], agg.to_numpy())


def main() -> None:
    t_start = time.time()
    lob = pd.read_parquet(LOB_IN)
    lob = (
        lob.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    )
    a1 = lob["BTCUSDT.ask_price_1"].to_numpy(np.float64)
    b1 = lob["BTCUSDT.bid_price_1"].to_numpy(np.float64)
    assert np.isfinite(a1).all() and np.isfinite(b1).all(), "NaN in level-1 prices"
    lob["mid_price"] = (a1 + b1) / 2.0
    for c in ("net_trade_sign", "trade_count", "ema_ofi"):  # loader wants these;
        lob[c] = 0.0  # Exp 4 never reads them
    lob.to_parquet(LOB_OUT, index=False)
    grid = lob["timestamp"].to_numpy(np.float64)
    print(f"derived LOB written: {len(lob)} rows -> {LOB_OUT}", flush=True)

    sell_min = np.full(len(grid), np.nan)
    buy_max = np.full(len(grid), np.nan)

    # 1) existing tape for its covered days
    tape = pd.read_parquet(TICKS_IN, columns=["timestamp", "price", "side"])
    accumulate(
        tape["timestamp"].to_numpy(np.float64),
        tape["price"].to_numpy(np.float64),
        tape["side"].to_numpy(np.int32),
        grid,
        sell_min,
        buy_max,
    )
    print(f"tape folded in: {len(tape)} prints", flush=True)

    # 2) overlap validation BEFORE trusting downloads: rebuild the check day
    #    from aggTrades alone and compare to the tape-derived extremes.
    chk = fetch_aggtrades(OVERLAP_CHECK)
    v_sell = np.full(len(grid), np.nan)
    v_buy = np.full(len(grid), np.nan)
    accumulate(
        chk["timestamp"].to_numpy(),
        chk["price"].to_numpy(),
        chk["side"].to_numpy(),
        grid,
        v_sell,
        v_buy,
    )
    day_lo = pd.Timestamp(OVERLAP_CHECK, tz="UTC").timestamp()
    in_day = (grid > day_lo) & (grid <= day_lo + 86400)
    for name, a, b in (("sell_min", sell_min, v_sell), ("buy_max", buy_max, v_buy)):
        both = in_day & np.isfinite(a) & np.isfinite(b)
        eq = np.isclose(a[both], b[both], rtol=0, atol=1e-9)
        print(
            f"overlap {OVERLAP_CHECK} {name}: {both.sum()} intervals, "
            f"{eq.mean():.4%} identical, max |diff| "
            f"{np.abs(a[both]-b[both]).max():.6f}",
            flush=True,
        )
        if eq.mean() < 0.999:
            sys.exit(f"VALIDATION FAILED on {name}: sources disagree, aborting")

    # 3) missing days from Binance aggTrades
    missing = [d for d in day_range(DAY0, DAY1) if not TAPE0 <= d <= TAPE1]
    for d in missing:
        df = fetch_aggtrades(d)
        accumulate(
            df["timestamp"].to_numpy(),
            df["price"].to_numpy(),
            df["side"].to_numpy(),
            grid,
            sell_min,
            buy_max,
        )
        print(f"  {d}: {len(df)} prints folded", flush=True)

    # 4) emit extremes rows keyed to exact grid timestamps
    rows = []
    for arr, side in ((sell_min, -1), (buy_max, 1)):
        w = np.isfinite(arr)
        rows.append(
            pd.DataFrame(
                {"timestamp": grid[w], "price": arr[w], "side": np.int32(side)}
            )
        )
    out = pd.concat(rows).sort_values("timestamp").reset_index(drop=True)
    out.to_parquet(EXTREMES_OUT, index=False)
    cov = np.isfinite(sell_min) | np.isfinite(buy_max)
    print(
        f"extremes written: {len(out)} rows -> {EXTREMES_OUT}; "
        f"interval coverage {cov.mean():.2%} of {len(grid)} rows; "
        f"total {time.time()-t_start:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
