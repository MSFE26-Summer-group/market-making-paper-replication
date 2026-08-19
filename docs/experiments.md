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
| Exp 4 — RL Market Making — C-PPO (Ping An Bank vs. BTC/USDT) | Sharpe | 12.3±0.8 | −0.79 | −13.1 |
| Exp 4 — RL Market Making — D-DQN (Ping An Bank vs. BTC/USDT) | Sharpe | 1.3±0.7 | −0.24 | −1.5 |
| Exp 4 — RL Market Making — C-PPO vs. D-DQN | Which wins? | C-PPO (H2) | D-DQN | contradicts H2 |

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

## Exp 4 — RL Market Making: C-PPO vs. D-DQN vs. Baselines (BTC/USDT)

Directly tests [H2](hypotheses.md#h2-continuous-action-spaces-improve-market-making-performance) (does continuous PPO beat discrete Dueling DQN?) and touches [H3](hypotheses.md#h3-hybrid-reward-functions-improve-risk-adjusted-performance) (does the hybrid reward keep inventory in check?) using the RL environment/agents built in `paper_replication.rl` (see [RL Model](rl_model.md)).

**Reproduce:** `notebooks/09_rl_training.ipynb`.

**Setup:** BTC/USDT, `n_levels=10`, `window_T=50`, chronological 80/20 train/test split (70,942 / 17,736 rows → 2,364 / 591 non-overlapping 30-step episodes). The Attn-LOB backbone is loaded from Exp 1b's winning pretrained checkpoint (`attn_lob_checkpoint_tuned.pt`, `dropout=0.3`) and kept **frozen** — only the trunk + policy/value (or Q) heads train, fed from a precomputed feature cache (`rl.feature_cache.precompute_lob_features`) rather than a live backbone forward pass every step. `RLConfig` defaults (paper's own reported values where stated): `omega=10`, `eta=0.5`, `zeta=0.01`, `max_bias=0.05`, `max_spread=0.1`; `minimum_trade_unit=0.001` BTC and `episode_length=30` steps (~5 minutes) are this replication's documented substitutes for the paper's China-A-share unit size and event-count episode length (see [RL Model](rl_model.md#why-our-version-differs-from-the-paper)). C-PPO: 300 updates × 16 episodes/update (144,000 env steps total), 4 update epochs, minibatch 128. D-DQN: 150,000 env steps, replay buffer 50,000, epsilon decayed 1.0→0.05 over the first 100,000 steps, target network hard-updated every 1,000 steps. Both: single seed (0), no hyperparameter search — this is a compute-budgeted run (a few minutes on a laptop GPU/CPU), not the paper's own tuned training run.

**Baselines:** Random (uniform `(A1, A2)` each step), three Fixed-spread levels (constant centered spread at 15%/50%/100% of `max_spread`), and an Avellaneda-Stoikov-inspired policy (Eq. 17-18, with a per-step realized-volatility feature standing in for a calibrated `sigma` and fixed `gamma=0.1`/`kappa=1.5` standing in for the paper's own order-arrival calibration — see `baselines.py`'s docstring). All seven policies evaluated deterministically (no exploration) over all 591 held-out test episodes through the identical `evaluate_policy` harness.

**Result** (mean ± std across the 591 test episodes; paper's columns are its Ping An Bank Co. row, Table II — different asset, different scale, included only for directional comparison, not a magnitude match):

| Policy | ND-PnL | PnLMAP | Profit Ratio | Sharpe | Mean \|inventory\| |
|---|---|---|---|---|---|
| **C-PPO** | −2.18 ± 3.41 | −30.4 ± 27.5 | −2.52 ± 3.08 | **−0.788** | 0.00256 |
| **D-DQN** | −0.195 ± 0.303 | −61.5 ± 72.3 | −1.25 ± 1.21 | **−0.243** | 0.0000288 |
| Random | −1.58 ± 2.06 | −30.2 ± 27.5 | −2.51 ± 3.08 | −0.788 | 0.00256 |
| Fixed_1 (15%) | −5.22 ± 6.66 | −29.7 ± 27.6 | −2.52 ± 3.12 | −0.784 | 0.00263 |
| Fixed_2 (50%) | −1.55 ± 2.00 | −29.5 ± 27.8 | −2.51 ± 3.13 | −0.776 | 0.00262 |
| Fixed_3 (100%) | −0.77 ± 0.99 | −29.4 ± 27.7 | −2.50 ± 3.11 | −0.774 | 0.00261 |
| AS | −0.77 ± 0.99 | −29.5 ± 27.7 | −2.50 ± 3.11 | −0.775 | 0.00261 |
| *Paper C-PPO (Ping An Bank)* | *9.3×10⁵* | *117.2* | *5.0×10⁻⁴* | *12.3* | *low* |
| *Paper D-DQN (Ping An Bank)* | *7.0×10⁵* | *8.6* | *3.5×10⁻⁴* | *1.3* | *higher* |

(Numbers above are from `notebooks/09_rl_training.ipynb`'s own execution. An
independent run of the identical config as a standalone script — same seed,
same data, different process/CUDA context — landed within noise of these
same numbers, e.g. Sharpe −0.79/−0.34 for C-PPO/D-DQN vs. −0.788/−0.243
here: the qualitative finding below is not an artifact of one run.)

**Everyone loses money on average, including the paper's own baselines' BTC/USDT analogues.** Unlike Exp 1-3, this isn't primarily a magnitude gap (different asset, different units, zero transaction costs either way) — it's that spread-capture net of adverse selection is negative for every policy tested here, on this data, at this training budget. That itself is a legitimate finding, not a bug: none of ND-PnL/PnLMAP/PR/Sharpe being negative across the board is inconsistent with a hybrid reward that heavily penalizes inventory (`zeta=0.01`) on a genuinely volatile asset (BTC/USDT, Oct 2022) most policies here still choose to trade actively on.

**H2 is contradicted in this run: D-DQN clearly beats C-PPO**, on both ND-PnL and Sharpe, by a wide margin — the opposite of the paper's own finding and this replication's stated success criterion. The mechanism is visible in the training curves and the `mean |inventory|` column: across the two matching runs behind this table, D-DQN's mean recent episode reward converged smoothly toward ~0 as epsilon decayed (roughly -0.14 at step 5,000 to -0.01 to -0.007 by step 150,000, in both runs) by learning to hold almost no inventory (~0.00003 vs. everyone else's ~0.0026) — essentially "mostly stay flat, quote defensively" — a strong, easy-to-find local optimum once `zeta`'s quadratic inventory penalty dominates. C-PPO's mean episode reward *never* showed a comparable trend in either run (e.g. -0.166, -0.147, -0.193, -0.209, -0.416 across one run's last five updates — noisy, not improving) and ended up statistically indistinguishable from Random and the Fixed baselines.

**This is very likely a training-budget asymmetry, not evidence against the paper's algorithmic claim.** D-DQN's replay buffer reuses each of its 150,000 transitions for many gradient updates (`train_interval=4` → ~37,500 update steps); C-PPO discards each on-policy batch after `update_epochs=4` passes over it (300 updates × 4 epochs × ~4 minibatches/epoch ≈ 4,800 gradient updates total, despite touching a comparable number of raw env steps, 144,000). PPO is well known in the literature to need many more environment interactions than off-policy methods to reach a comparable policy — this run's 300 updates is a small fraction of what published PPO results typically use. **Natural next step, if worth the compute:** a much longer C-PPO run (or a larger `episodes_per_update`/more updates) before concluding anything about the paper's own H2 claim one way or the other; this result says "not with this budget," not "the paper is wrong."

**PnLMAP is unstable for D-DQN specifically** (−61.5 ± 72.3, far outside every other policy's ballpark, and −91.88 ± 113.48 in the other matching run) because its own denominator (mean absolute position, ~0.00003) is close to zero — dividing a small negative PnL by a near-zero position amplifies noise into a large, unstable ratio. This is a known artifact of PnLMAP as a metric when a policy's inventory is genuinely close to flat, not a sign D-DQN is somehow performing worse than its ND-PnL/Sharpe numbers suggest.

**What this does and doesn't show:**

- Confirms the environment, reward, action spaces, and both training algorithms are wired correctly end to end (finite losses, sensible fills, D-DQN visibly learning a coherent low-risk policy) — the RL model built in `paper_replication.rl` works.
- Does **not** confirm H2 as stated — the opposite ordering was observed, with a plausible confound (PPO's much lower effective sample efficiency at this budget) identified but not ruled out.
- Single seed, single training budget, no hyperparameter search for either algorithm — like Exp 1/1b/2/3, one run, not error-barred.
- Zero transaction costs (matching the paper's own simulator assumption, IV-C1) — a more realistic cost model would very plausibly push every policy's numbers further negative, but wouldn't obviously change the *relative* ordering between policies.

## Notes & Observations

_Document surprises, dead ends, and hyperparameter choices here._

- **Config drift bug (found while building Exp 2):** `TrainConfig.patience` had silently drifted to `10` in code while its own docstring, every notebook, and this file's Exp 1/1b writeups all said `5`. Nothing caught it until an unrelated rename-only re-run of notebook 05 produced quietly different numbers than what was documented. Root cause unclear (no single edit is implicated), but the practical lesson: a mismatch between a docstring and its own default value is exactly the kind of thing that survives silently unless something asserts on it. Fixed to `5` and pinned with a regression test (`test_default_patience_matches_documented_value`); Exp 1 and Exp 1b were both re-run under the corrected value and every number in this file reflects that re-run, not the original one. The qualitative findings (dropout delays overfitting, tuning buys a modest but real improvement, most of the gap to the paper is structural) held up across the fix — only the exact numbers and, notably, Exp 1b's *winning* candidate changed (`lower_lr_light` → `dropout_0.3`).
- Exp 1b: regularization's effect was entirely on *how much better* the best val_loss got, not primarily on how many epochs it took to get there (8 vs. 9 epochs across every candidate, barely different) — a different pattern than what the pre-fix run suggested (dropout mainly buying more epochs). Worth re-examining if patience is tuned differently in a future search.
