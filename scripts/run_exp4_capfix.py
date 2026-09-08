"""Step A of the scale-transfer ladder: correct ONLY the caps, nothing else.

max_bias/max_spread go from the literal USD transfer ($0.05/$0.10, i.e.
0.19x the BTC market spread) to the paper's own invariant — the cap as a
multiple of the market spread (10 ticks ~= 5-10x spread on all three of
its stocks). BTC spread ~$0.54 -> max_spread = $5.40 (10x), max_bias =
$2.70 (paper's 1:2 bias:spread ratio). zeta stays 0.01 USD (Step B later).

Two windows, both directly comparable to the clean-window suite:
  capfix_10d  = run2a's exact config + new caps (train 10-19..10-28 18:37,
                test 10-28 18:37..10-30)
  capfix_29d  = run1_clean29's exact config + new caps (10-10..11-07, 80/20)

    PYTHONPATH=src .venv/bin/python scripts/run_exp4_capfix.py
"""

import json
import platform
import time
from pathlib import Path

import torch

from paper_replication.rl.tickfill.exp4 import run_exp4

LOB = "data/btc_usdt_20221010_20221110_lob_derived.parquet"
TICKS = "data/btc_usdt_20221010_20221110_tick_extremes.parquet"
CKPT = "results/exp1b_attn_lob.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MAX_BIAS = 2.70  # USD; 5x market spread (paper: cap ~5-10x spread)
MAX_SPREAD = 5.40  # USD; 10x market spread

TS_1010 = 1665360000.0
TS_1019 = 1666137600.0
TS_1031 = 1667174400.0
TS_1108 = 1667865600.0
SPLIT_11D = 1666982260.0

RUNS = {
    "capfix_10d": dict(ts_min=TS_1019, ts_max=TS_1031, split_ts=SPLIT_11D),
    "capfix_29d": dict(ts_min=TS_1010, ts_max=TS_1108),
}

for name, kw in RUNS.items():
    out = f"results/exp4_{name}"
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    results = run_exp4(
        LOB,
        CKPT,
        out,
        device=DEVICE,
        fill_mode="tick",
        ticks_parquet=TICKS,
        max_bias=MAX_BIAS,
        max_spread=MAX_SPREAD,
        **kw,
    )
    elapsed = time.time() - t0
    meta = {
        "run": name,
        "fills": "tick",
        "max_bias": MAX_BIAS,
        "max_spread": MAX_SPREAD,
        "zeta": "0.01 (unchanged, Step A)",
        "params": dict(kw),
        "device": DEVICE,
        "machine": platform.platform(),
        "total_seconds": round(elapsed, 1),
    }
    Path(out, "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{name} done in {elapsed:.0f}s", flush=True)
    for pol, m in results.items():
        print(
            f"  {pol:<12} nd={m['nd_pnl_mean']:>9.4f}±{m['nd_pnl_std']:<8.4f}"
            f" sharpe={m['sharpe']:>7.3f} pnl$={m['pnl_mean_usd']:>9.5f}",
            flush=True,
        )
