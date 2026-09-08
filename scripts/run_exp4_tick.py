"""Exp 4 single-variable rerun: identical setup, fills switched to tick side-aware.

PYTHONPATH=src .venv/bin/python scripts/run_exp4_tick.py
"""

import time

import torch

from paper_replication.rl.tickfill.exp4 import run_exp4

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
t0 = time.time()
results = run_exp4(
    "data/btc_usdt_20221019_20221030_lob.parquet",
    "results/exp1b_attn_lob.pt",
    "results/exp4_tick",
    device=DEVICE,
    fill_mode="tick",
    ticks_parquet="data/btc_usdt_20221019_20221030_ticks.parquet",
)
print(f"\nExp 4 (tick fills) done in {time.time()-t0:.0f}s\n")
hdr = f"{'policy':<12}{'ND-PnL':>15}{'PnLMAP':>17}{'PR':>12}{'Sharpe':>9}{'PnL$':>10}"
print(hdr)
print("-" * len(hdr))
for name, m in results.items():
    print(
        f"{name:<12}{m['nd_pnl_mean']:>9.2f}±{m['nd_pnl_std']:<5.2f}"
        f"{m['pnlmap_mean']:>10.1f}±{m['pnlmap_std']:<6.1f}"
        f"{m['profit_ratio_mean']:>12.5f}{m['sharpe']:>9.3f}{m['pnl_mean_usd']:>10.4f}"
    )
