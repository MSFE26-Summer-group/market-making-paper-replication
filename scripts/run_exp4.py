"""Run Experiment 4 end to end: pretrain (if needed) -> RL -> evaluation.

PYTHONPATH=src .venv/bin/python scripts/run_exp4.py
"""

import time
from pathlib import Path

import torch

from paper_replication.features.config import FeatureConfig
from paper_replication.features.dataset import (
    build_feature_dataset,
    chronological_split,
)
from paper_replication.models.attn_lob import AttnLOB, AttnLOBConfig
from paper_replication.rl.tickfill.exp4 import run_exp4
from paper_replication.training.config import TrainConfig
from paper_replication.training.loop import train_model

LOB = "data/btc_usdt_20221019_20221030_lob.parquet"
CKPT = "results/exp1b_attn_lob.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

if not Path(CKPT).exists():
    print(f"pretraining Attn-LOB (Exp 1b winning config) on {DEVICE}...", flush=True)
    t0 = time.time()
    ds = build_feature_dataset(LOB, FeatureConfig())
    splits = chronological_split(ds, test_frac=0.5, val_frac_of_train=0.2)
    print(
        f"dataset: {len(splits['train'])} train / {len(splits['val'])} val", flush=True
    )
    model = AttnLOB(AttnLOBConfig(dropout=0.3))
    cfg = TrainConfig(
        batch_size=256,
        epochs=50,
        patience=5,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=0,
        device=DEVICE,
        checkpoint_path=CKPT,
    )
    hist = train_model(model, splits["train"], splits["val"], cfg, num_classes=3)
    print(
        f"pretrained in {time.time()-t0:.0f}s, {len(hist)} epochs, "
        f"best val_loss {min(h['val_loss'] for h in hist):.4f}",
        flush=True,
    )

t0 = time.time()
results = run_exp4(LOB, CKPT, "results/exp4", device=DEVICE)
print(f"\nExp 4 done in {time.time()-t0:.0f}s\n")
hdr = f"{'policy':<12}{'ND-PnL':>14}{'PnLMAP':>16}{'PR':>12}{'Sharpe':>9}"
print(hdr)
print("-" * len(hdr))
for name, m in results.items():
    print(
        f"{name:<12}{m['nd_pnl_mean']:>8.2f}±{m['nd_pnl_std']:<5.2f}"
        f"{m['pnlmap_mean']:>9.1f}±{m['pnlmap_std']:<6.1f}"
        f"{m['profit_ratio_mean']:>12.5f}{m['sharpe']:>9.3f}"
    )
