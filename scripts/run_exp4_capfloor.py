"""Step A': corrected caps + the paper's implicit spread floor.

The paper has no explicit min-spread hyperparameter (Eq. 10 allows
spread = 0), but two things make a floor implicit in its market: the
Fixed baselines quote at LOB levels 1-3 (level 1 = the touch, never
inside), and the A-share tick (~1 tick = the whole market spread) leaves
no room inside the touch anyway. BTC's relatively 1000x finer tick opens
that region up, so the mirror must add it back explicitly: quotes are
clamped to the current best bid/ask (TOUCH_FLOOR). zeta stays 0.01 (Step
B later). Same two windows as run_exp4_capfix.py for direct comparison.

    PYTHONPATH=src .venv/bin/python scripts/run_exp4_capfloor.py
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

RUNS = {
    "capfloor_10d": dict(
        ts_min=1666137600.0, ts_max=1667174400.0, split_ts=1666982260.0
    ),
    "capfloor_29d": dict(ts_min=1665360000.0, ts_max=1667865600.0),
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
        max_bias=2.70,
        max_spread=5.40,
        touch_floor=True,
        **kw,
    )
    elapsed = time.time() - t0
    meta = {
        "run": name,
        "fills": "tick",
        "max_bias": 2.70,
        "max_spread": 5.40,
        "touch_floor": True,
        "zeta": "0.01 (unchanged)",
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
