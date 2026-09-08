"""Exp 4 clean-window expansion suite (tick fills throughout).

Four runs on the monthly file's grid, motivated by the daily-RV scan
showing the FTX regime break is exactly 11-08..11-10 (RV jumps 2-3.7%
-> 8-8.6%); everything through 11-07 23:59 is statistically normal:

  run1        10-10..11-07, 80/20 chronological split  (main result)
  run1_budget same, training budget scaled ~2.6x with the data
  run2a       train 10-19..10-28 18:37, test 10-28 18:37..10-30 (re-grid
              baseline: the 11-day run's calendar windows on this grid)
  run2b       train 10-10..10-28 18:37, same fixed test (single variable:
              ~2x training data, identical referee period)

    PYTHONPATH=src .venv/bin/python scripts/run_exp4_clean_window.py
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

TS_1010 = 1665360000.0  # 2022-10-10 00:00 UTC
TS_1019 = 1666137600.0  # 2022-10-19 00:00
TS_1031 = 1667174400.0  # 2022-10-31 00:00 (end of 11-day file's calendar)
TS_1108 = 1667865600.0  # 2022-11-08 00:00 (FTX break; exclusive end)
SPLIT_11D = 1666982260.0  # 2022-10-28 18:37:40, the 11-day run's boundary

RUNS = {
    "run1_clean29": dict(ts_min=TS_1010, ts_max=TS_1108),
    "run1_clean29_budget": dict(
        ts_min=TS_1010, ts_max=TS_1108, cppo_updates=780, ddqn_steps=390_000
    ),
    "run2a_fixedtest_9d": dict(ts_min=TS_1019, ts_max=TS_1031, split_ts=SPLIT_11D),
    "run2b_fixedtest_18d": dict(ts_min=TS_1010, ts_max=TS_1031, split_ts=SPLIT_11D),
}

for name, kw in RUNS.items():
    out = f"results/exp4_{name}"
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    results = run_exp4(
        LOB, CKPT, out, device=DEVICE, fill_mode="tick", ticks_parquet=TICKS, **kw
    )
    elapsed = time.time() - t0
    meta = {
        "run": name,
        "fills": "tick",
        "params": {k: v for k, v in kw.items()},
        "device": DEVICE,
        "machine": platform.platform(),
        "total_seconds": round(elapsed, 1),
    }
    Path(out, "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{name} done in {elapsed:.0f}s", flush=True)
    for pol, m in results.items():
        print(
            f"  {pol:<12} nd={m['nd_pnl_mean']:>8.4f}±{m['nd_pnl_std']:<7.4f}"
            f" sharpe={m['sharpe']:>7.3f} pnl$={m['pnl_mean_usd']:>8.4f}",
            flush=True,
        )
