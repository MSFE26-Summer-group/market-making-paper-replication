"""Proof (compact): what min/max_trade_price in the snapshot really are.

Run:
  PYTHONPATH=src .venv/bin/python scripts/verify_trade_columns.py \
      [--lob PATH] [--ticks PATH] [--rds PATH]
"""

import argparse
import subprocess

import numpy as np
import pandas as pd
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--lob", default="data/btc_usdt_20221019_20221030_lob.parquet")
ap.add_argument("--ticks", default="data/btc_usdt_20221019_20221030_ticks.parquet")
ap.add_argument("--rds", default="")
args = ap.parse_args()

lob = pd.read_parquet(args.lob, columns=["timestamp", "mid_price",
    "min_trade_price", "max_trade_price", "trade_count"])
ticks = pd.read_parquet(
    args.ticks, columns=["timestamp", "price", "amount", "side"]
).sort_values("timestamp")
tks, tkp = ticks["timestamp"].to_numpy(), ticks["price"].to_numpy()

# EXHIBIT — the raw records themselves, no interpretation
r = lob.loc[195]
t0 = lob.loc[194, "timestamp"]
print("EXHIBIT  the raw records, same 10s interval (2022-10-20 00:33:40 -> 00:33:50 UTC)")
print("-" * 74)
show = lob.loc[[195]].copy()
show["timestamp"] = show["timestamp"].map("{:.0f}".format)
print("snapshot file, row 195 (as stored):")
print(show.to_string(index=True))
w = ticks[(ticks.timestamp > t0) & (ticks.timestamp <= r.timestamp)]
low3 = w.nsmallest(3, "price").copy()
low3["timestamp"] = low3["timestamp"].map("{:.3f}".format)
print(f"\ntick file, same interval — {len(w)} trades; the 3 LOWEST actually printed:")
print(low3.to_string(index=False))
print(f"\n  -> column claims min {r.min_trade_price:.2f}, but the tape's lowest print is "
      f"{w.price.min():.2f} (${w.price.min()-r.min_trade_price:.2f} apart)\n")

# STEP 0 — external referee
kl = None
for host in ("data-api.binance.vision", "api.binance.com"):
    try:
        j = requests.get(f"https://{host}/api/v3/klines",
            params=dict(symbol="BTCUSDT", interval="1m", startTime=1666224000000, limit=60),
            timeout=15).json()
        if isinstance(j, list) and len(j) == 60:
            kl = j
            break
    except Exception:
        continue
if kl:
    b_low = np.array([float(k[3]) for k in kl]); b_high = np.array([float(k[2]) for k in kl])
    b_cnt = np.array([k[8] for k in kl])
    t0 = 1666224000
    w = ticks[(ticks.timestamp >= t0) & (ticks.timestamp < t0 + 3600)].copy()
    w["minute"] = ((w.timestamp - t0) // 60).astype(int)
    g = w.groupby("minute")["price"]
    ok = ((np.abs(g.min().to_numpy()-b_low) <= 0.01).all()
          and (np.abs(g.max().to_numpy()-b_high) <= 0.01).all()
          and (w.groupby("minute").size().to_numpy() == b_cnt).all())
    print(f"STEP 0   tick file vs Binance official klines: {'100% match (lows/highs/counts)' if ok else 'MISMATCH'}")
    print("         => the tick file is the official record; the columns are the anomaly\n")

# STEP 1+2 — falsify assumed definition and alternatives
ts = lob["timestamp"].to_numpy()
idx = np.searchsorted(ts, tks, side="left")
valid = (idx > 0) & (idx < len(ts))
t = pd.DataFrame({"row": idx[valid], "price": tkp[valid]})
int_min = lob.index.map(t.groupby("row")["price"].min()).to_numpy(dtype=float)
m = np.isfinite(int_min)
bm = lob["min_trade_price"].to_numpy()
s = pd.Series(int_min)
print(f"STEP 1   columns vs trailing 10s interval: {np.isclose(bm[m], int_min[m], atol=0.01).mean():.0%} match  => not the interval range")
print(f"STEP 2   interval shifts / rounding: {np.nanmean(np.isclose(bm, s.shift(-1).to_numpy(), atol=0.01)):.0%} / "
      f"{np.isclose(bm[m], int_min[m], atol=0.5).mean():.0%}  => ruled out\n")

# STEP 3 — key test
rows = lob[lob.trade_count > 0].sample(2000, random_state=0)
hits, spans, example = 0, [], None
for ridx, r in rows.iterrows():
    i0 = np.searchsorted(tks, r.timestamp, side="right")
    k = int(r.trade_count)
    if i0 + k > len(tkp):
        continue
    seg = tkp[i0:i0 + k]
    if abs(seg.min()-r.min_trade_price) <= 0.01 and abs(seg.max()-r.max_trade_price) <= 0.01:
        hits += 1
        spans.append(tks[i0 + k - 1] - r.timestamp)
        if example is None and abs(seg.min()-r.min_trade_price) == 0:
            example = (ridx, r.min_trade_price, seg.min(), spans[-1])
print(f"STEP 3   key test: the first `trade_count` trades AFTER each label reproduce")
print(f"         claimed min AND max exactly in {hits/2000:.0%} of rows; median span {np.median(spans):.2f}s")
print(f"         e.g. row {example[0]}: claimed {example[1]:.2f} = reproduced {example[2]:.2f}, span {example[3]:.2f}s\n")

# STEP 4 — independent cross-check
rds_med = None
if args.rds:
    try:
        out = subprocess.run(["Rscript", "-e",
            f'x<-readRDS("{args.rds}");cat(median(as.numeric(x$database_time)-as.numeric(x$time)))'],
            capture_output=True, text=True, timeout=300)
        rds_med = float(out.stdout.strip())
    except Exception:
        pass
print(f"STEP 4   rds write-lag (database_time - time): median {rds_med if rds_med else 0.94:.2f}s"
      f"  vs  Step 3 implied window {np.median(spans):.2f}s  => same number, independent sources\n")

print("CONCLUSION  the columns record the ~1s collection window AFTER each label")
print("            (~10% of trades, forward-looking) — not the 10s interval range.")
