"""Exp 4 on the full monthly file (2022-10-10 .. 2022-11-10), tick fills.

Same configuration as scripts/run_exp4_tick.py — frozen Exp 1b backbone,
identical hyperparameters and budgets — only the data window changes
(88,727 -> ~276k snapshot rows, ~3.1x train/test episodes). Requires
scripts/build_monthly_tick_extremes.py to have been run first.

    PYTHONPATH=src .venv/bin/python scripts/run_exp4_month_tick.py
"""

import json
import platform
import time
from pathlib import Path

import torch

from paper_replication.rl.tickfill.exp4 import run_exp4

LOB = "data/btc_usdt_20221010_20221110_lob_derived.parquet"
TICKS = "data/btc_usdt_20221010_20221110_tick_extremes.parquet"
OUT = "results/exp4_month_tick"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

t0 = time.time()
results = run_exp4(
    LOB,
    "results/exp1b_attn_lob.pt",
    OUT,
    device=DEVICE,
    fill_mode="tick",
    ticks_parquet=TICKS,
)
elapsed = time.time() - t0
meta = {
    "data": LOB,
    "fills": "tick",
    "device": DEVICE,
    "machine": platform.platform(),
    "total_seconds": round(elapsed, 1),
}
Path(OUT, "run_meta.json").write_text(json.dumps(meta, indent=2))
print(f"\nExp 4 monthly (tick fills) done in {elapsed:.0f}s\n")
hdr = f"{'policy':<12}{'ND-PnL':>15}{'PnLMAP':>17}{'PR':>12}{'Sharpe':>9}{'PnL$':>10}"
print(hdr)
print("-" * len(hdr))
for name, m in results.items():
    print(
        f"{name:<12}{m['nd_pnl_mean']:>9.2f}±{m['nd_pnl_std']:<5.2f}"
        f"{m['pnlmap_mean']:>10.1f}±{m['pnlmap_std']:<6.1f}"
        f"{m['profit_ratio_mean']:>12.5f}{m['sharpe']:>9.3f}{m['pnl_mean_usd']:>10.4f}"
    )
