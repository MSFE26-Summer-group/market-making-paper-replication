# Experiments

## Running Experiments

```bash
python -m paper_replication.experiments.exp1
```

## Results

| Experiment | Metric | Paper Result | Our Result | Delta |
|------------|--------|-------------|-----------|-------|
| Exp 1 — Attn-LOB Pretrain Classification | Precision | 0.7663 | 0.4966 | −0.2697 |
| Exp 1 — Attn-LOB Pretrain Classification | Recall | 0.7019 | 0.4943 | −0.2076 |
| Exp 1 — Attn-LOB Pretrain Classification | F1 | 0.7284 | 0.4948 | −0.2336 |
| Exp 1 — Attn-LOB Pretrain Classification | Accuracy | — (not reported) | 0.5166 | — |
| Exp 1b — Attn-LOB Pretrain Classification (tuned) | Precision | 0.7663 | 0.5231 | −0.2432 |
| Exp 1b — Attn-LOB Pretrain Classification (tuned) | Recall | 0.7019 | 0.5224 | −0.1795 |
| Exp 1b — Attn-LOB Pretrain Classification (tuned) | F1 | 0.7284 | 0.5224 | −0.2060 |
| Exp 1b — Attn-LOB Pretrain Classification (tuned) | Accuracy | — (not reported) | 0.5465 | — |
| Exp 2 — Baseline Comparison — AttnLOB | F1 | 0.7284 | 0.4963 | −0.2321 |
| Exp 2 — Baseline Comparison — FCLOB | F1 | 0.5660 | 0.5199 | −0.0461 |
| Exp 2 — Baseline Comparison — ConvLOB | F1 | 0.4984 | 0.5425 | +0.0441 |
| Exp 2 — Baseline Comparison — DeepLOB | F1 | 0.7118 | 0.5392 | −0.1726 |
| Exp 3 — Attn-LOB Depth Ablation — n_levels=10 (Oct-20 only) | F1 | — (not a paper config) | 0.4411 | — |
| Exp 3 — Attn-LOB Depth Ablation — n_levels=50 (Oct-20 only) | F1 | — (not a paper config) | 0.4679 | — |

## Exp 1 — Attn-LOB Pretrain Classification (BTC/USDT)

Supports [H1](hypotheses.md#h1-automatic-lob-representation-learning-improves-market-making) as a precursor step: before Attn-LOB's learned representation can be evaluated on trading performance (ND-PnL/PnLMAP/Sharpe), it first needs to work as a mid-price direction classifier (paper Table I). This experiment is that classification benchmark, not a direct test of H1 itself.

**Reproduce:** `notebooks/05_attn_lob_training.ipynb` (architecture: [Attn-LOB Model](attn_lob_model.md); dataset: [Feature Engineering Pipeline](feature_engineering.md)).

**Setup:** BTC/USDT, `window_T=50`, `n_levels=10`, chronological train/val/test split (35,467 / 8,867 / 44,334 samples), Adam (`lr=1e-3`), `batch_size=256`, early stopping (`patience=5`) — stopped at epoch 8, kept epoch 3's weights (lowest val_loss). No learning-rate schedule, no weight_decay/dropout, no hyperparameter search. Paper's row above is their Attn-LOB result on Ping An Bank Co. (Table I); for scale, their weakest baseline (Conv-LOB) scored F1=0.4984 and their strongest (their own Attn-LOB) scored F1=0.7284.

**Reading the gap:** our F1 (0.4948) landed just below the paper's *weakest* baseline, about 0.23 below their Attn-LOB. Three things are conflated in that gap and this run doesn't separate them:

1. **Untuned hyperparameters** — no LR schedule, no regularization, no search over batch size/learning rate, a handful of early-stopped epochs. This is the one an actual tuning pass would address.
2. **Different asset/data structure** — BTC/USDT crypto vs. their Chinese equities, and critically our data is a ~10s snapshot grid rather than raw ~60-150ms order-book events (see [Feature Engineering Pipeline](feature_engineering.md#why-our-version-differs-from-the-paper)). `window_T=50` is therefore ~8 minutes of context here vs. a few seconds in the paper — a structurally different prediction problem, not just a harder version of the same one.
3. **Auto-calibrated label threshold** — our `alpha` is fit from the data to balance classes (see [labels.py](feature_engineering.md#labelspy-the-answer-key)) rather than reusing the paper's `alpha=1e-5`, which doesn't transfer across assets anyway.

(1) is fixable by tuning. (2) and (3) are not — they're consequences of what data is actually available to this replication, and tuning won't close that part of the gap.

## Exp 1b — Attn-LOB Pretrain Classification, Tuned (BTC/USDT)

Direct follow-up to Exp 1's three-way gap breakdown: this run addresses (1) — untuned hyperparameters — and leaves (2) and (3) untouched, to see how much of the gap was actually hyperparameters.

**Reproduce:** `notebooks/06_attn_lob_hyperparameter_search.ipynb`.

**Setup:** same data/split as Exp 1. An 8-candidate search over `dropout` (0/0.1/0.3), `weight_decay` (0/1e-4/1e-3), and one lower learning rate (5e-4 vs. 1e-3), each selected on `val_loss` only via `paper_replication.training.search.run_search` — the test split was never touched during the search, only once at the end for the winning candidate. `baseline` (the search's own re-run of Exp 1's exact config) reproduced val_loss=0.9749, close to Exp 1's own run — a useful sanity check that the search harness itself isn't the source of any difference below.

**Result:** winner was `dropout_0.3` (`lr=1e-3`, `weight_decay=0`, `dropout=0.3`), val_loss=0.9316 vs. baseline's 0.9749. Every regularized or lower-LR candidate beat the baseline; dropout alone at a fairly high rate outperformed every combination that also used weight_decay or a lower learning rate.

Test-set F1 moved from 0.4948 (Exp 1) to 0.5224 (Exp 1b) — a consistent ~2.5-3 point improvement across precision/recall/F1/accuracy, confirming untuned hyperparameters were genuinely *part* of the gap to the paper (0.7284), but a modest part. The remaining ~0.21 F1 points align with Exp 1's (2) and (3) — asset/sampling-grid differences and the auto-calibrated label threshold — which this tuning pass was never going to touch, by design.

## Exp 2 — Baseline Comparison: AttnLOB vs. FCLOB vs. ConvLOB vs. DeepLOB (BTC/USDT)

Directly tests [H1](hypotheses.md#h1-automatic-lob-representation-learning-improves-market-making)'s premise that Attn-LOB's learned representation outperforms the alternative architectures the paper compares it against (Table I). All four models — [Attn-LOB](attn_lob_model.md), FC-LOB, Conv-LOB, DeepLOB (see each model's module docstring for architecture details and paper-fidelity notes) — trained on identical data/split with identical *untuned* defaults, so differences reflect architecture, not who got a better hyperparameter search. Deliberately compared against Exp 1's untuned Attn-LOB number, not Exp 1b's tuned one — tuning only one of the four would stack the deck.

**Reproduce:** `notebooks/07_baseline_comparison.ipynb`.

**Setup:** same data/split as Exp 1/1b. Each model's own default config; shared `TrainConfig` (`batch_size=256`, `lr=1e-3`, `weight_decay=0`, `patience=5`, `seed=0`, device auto-selected).

**Result:**

| Model | Params | Epochs | Best val_loss | Test F1 | Paper F1 | Delta |
|---|---|---|---|---|---|---|
| ConvLOB | 20,035 | 12 | 0.9051 | 0.5425 | 0.4984 | +0.0441 |
| DeepLOB | 115,043 | 7 | 0.8990 | 0.5392 | 0.7118 | −0.1726 |
| FCLOB | 2,328,067 | 8 | 0.9279 | 0.5199 | 0.5660 | −0.0461 |
| AttnLOB | 197,987 | 9 | 0.9762 | 0.4963 | 0.7284 | −0.2321 |

**The ordering is completely inverted from the paper.** Paper: AttnLOB > DeepLOB > FCLOB > ConvLOB. Ours: ConvLOB > DeepLOB > FCLOB > AttnLOB. The paper's best-reported architecture is our worst performer; its worst-reported baseline (Conv-LOB, by far the smallest model here at 20K params) is our best.

This isn't noise, and it isn't arbitrary either: `AttnLOB`'s `dropout` defaults to `0.0` (the paper's Fig. 1 doesn't specify a value), and Exp 1b already showed `dropout=0.3` alone moves its F1 from 0.4948 to 0.5224 on this exact data. None of the other three baselines received any equivalent regularization pass here either, but self-attention is the highest-capacity, most overfitting-prone component of the four architectures, so it's the one that suffers most from having none. Conv-LOB's causal dilated-residual stack, at a fifth of DeepLOB's parameter count and a hundredth of FC-LOB's, simply has far less room to overfit 35K training samples in the first place — plausibly why it wins here, not because it's a categorically "better" architecture.

**What this does and doesn't show:**

- **Not evidence against the paper's finding.** The paper's numbers presumably came from tuned models; this is untuned-vs-untuned. The one model we *have* tuned so far (AttnLOB, Exp 1b) already closed a meaningful chunk of its gap to the others with nothing but dropout.
- **Not a fully stable ranking.** Best-val_loss ordering (DeepLOB < ConvLOB < FCLOB < AttnLOB) and best-test-F1 ordering (same, reading ascending) already disagree slightly on DeepLOB vs. ConvLOB — expected single-split noise, not a red flag.
- **Directly fails one of this replication's stated success criteria** ([Replication Framework](replication_framework.md)'s "reproduced metrics follow the same relative ordering reported by the authors") — at least in this untuned form. Worth stating plainly rather than downplaying.

**Natural next step**, if worth the compute: repeat this comparison with an equivalent tuning pass (dropout/weight_decay/LR search, matching Exp 1b's methodology) applied to all four models before ranking them. That would distinguish "the paper's ordering re-emerges once every model gets a fair shot" from "this dataset structurally favors simpler architectures regardless of tuning."

## Exp 3 — Attn-LOB LOB-Depth Ablation (BTC/USDT, 2022-10-20 only)

Follow-up to Exp 2's gap: could Attn-LOB close some of the distance to the paper with **more data**? We checked `rds_transformed.parquet` and its source R2 bucket as a way to extend training data specifically for Attn-LOB.

**Finding: no more data exists.** `rds_transformed.parquet` (and the R2 bucket it came from) covers only 2022-10-20 — 8,127 snapshots — and that day is already the *first* day inside `btc_usdt_20221019_20221030_lob.parquet`, the file every prior experiment already trains on. Confirmed directly: `BTCUSDT.ask_price_1..10` in the existing multi-day file matches `rds_transformed.parquet`'s best bid/ask row-for-row across all 8,127 rows. There is no new *time* available in this data, in the repo, or in the R2 bucket (whose full listing has exactly two objects, both for this same day).

What that file *does* have that's never been used: full order-book depth (thousands of price levels per side vs. the 10 our pipeline has always used). So this experiment tests depth instead of volume: does giving Attn-LOB the top 50 levels instead of top 10, on the exact same single day/samples/split/hyperparameters, change anything? A required side effect: `layers.ConvBlock` (shared by AttnLOB and DeepLOB) previously hard-required `n_levels=10` (`NotImplementedError` otherwise) — generalized to support any `n_levels` that's a multiple of 5, so this ablation is possible. `n_levels=50` is a deliberate deviation from the paper's Fig. 1, used only for this ablation.

**Reproduce:** `notebooks/08_lob_depth_ablation.ipynb`.

**Setup:** both arms use the identical 8,068 windowed samples from 2022-10-20 (`window_T=50`, `horizon_k=10`, auto-calibrated `alpha`), chronological split (train=3,227 / val=807 / test=4,034), and Exp 1b's winning hyperparameters (`dropout=0.3`, `lr=1e-3`, `weight_decay=0`, `batch_size=256`, `patience=5`, `seed=0`) — `n_levels` is the only thing that differs between arms.

**Result:**

| Arm | Params | Epochs | Best val_loss | Test Precision | Test Recall | Test F1 | Test Accuracy |
|---|---|---|---|---|---|---|---|
| n_levels=10 | 197,987 | 10 | 1.1104 | 0.5072 | 0.4543 | 0.4411 | 0.4487 |
| n_levels=50 | 214,371 | 8 | 1.0444 | 0.5073 | 0.4697 | 0.4679 | 0.4695 |

Depth-50 vs. depth-10: val_loss improved (1.1104 → 1.0444) and test F1 improved (0.4411 → 0.4679, **+0.0269**), consistently in the same direction on both the selection metric and the held-out metric. The gain came entirely from recall/accuracy — precision was flat (0.5072 vs. 0.5073).

**But sample count dominates depth by a wide margin.** Both arms here (3,227 train samples, one day) land well below Exp 1b's F1=0.5224, which used the *same* `n_levels=10` architecture but eleven days (35,467 train samples). Depth alone bought about six points of relative F1 improvement; the missing ten days of data cost far more than that.

**What this does and doesn't show:**

- Answers the original "more data" question, just not the way expected: there isn't any more BTC/USDT LOB history available (repo or R2 bucket) beyond the eleven days already in use. This depth ablation is the closest available proxy for "give the model more information about what we have," and the answer is: it helps, modestly.
- Not a claim that `n_levels=50` is a better architecture in general — it's a deviation from the paper's own Fig. 1, used here only to answer this specific question. Not used anywhere else in this replication.
- Single split, single seed per arm — like Exp 1/1b/2, this is one run, not error-barred, though the consistent direction across val_loss and test F1 makes it unlikely to be pure noise.
- Natural next step, if worth the compute: an intermediate depth (e.g. `n_levels=20` or `30`) to see whether the F1 gain scales roughly linearly with depth or saturates quickly.

## Notes & Observations

_Document surprises, dead ends, and hyperparameter choices here._

- **Config drift bug (found while building Exp 2):** `TrainConfig.patience` had silently drifted to `10` in code while its own docstring, every notebook, and this file's Exp 1/1b writeups all said `5`. Nothing caught it until an unrelated rename-only re-run of notebook 05 produced quietly different numbers than what was documented. Root cause unclear (no single edit is implicated), but the practical lesson: a mismatch between a docstring and its own default value is exactly the kind of thing that survives silently unless something asserts on it. Fixed to `5` and pinned with a regression test (`test_default_patience_matches_documented_value`); Exp 1 and Exp 1b were both re-run under the corrected value and every number in this file reflects that re-run, not the original one. The qualitative findings (dropout delays overfitting, tuning buys a modest but real improvement, most of the gap to the paper is structural) held up across the fix — only the exact numbers and, notably, Exp 1b's *winning* candidate changed (`lower_lr_light` → `dropout_0.3`).
- Exp 1b: regularization's effect was entirely on *how much better* the best val_loss got, not primarily on how many epochs it took to get there (8 vs. 9 epochs across every candidate, barely different) — a different pattern than what the pre-fix run suggested (dropout mainly buying more epochs). Worth re-examining if patience is tuned differently in a future search.
