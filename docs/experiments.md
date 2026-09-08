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

---

## Run 1 — C-PPO vs benchmarks, snapshot fills (2026-07-23)

**Setup**: BTC-USDT 10s snapshots (Oct 20-30, 2022), 88,678 states.
Train = first 70%, test = last 30% (73 non-overlapping 1-hour episodes).
C-PPO: 500 updates x 4 episodes, 833s wall-clock on Apple Silicon (CPU).
Fill model: interval trade-price range crosses quote (no queue, no side).
Env: eta=0.5, zeta=0.01, max_inv=10, bias<=5bps, half-spread 0.5-10bps.

| Strategy | PnL/ep (USD/unit) | Sharpe | PnLMAP | ND-PnL | Fills/ep | Mean inv |
|---|---|---|---|---|---|---|
| C-PPO (trained) | **-13.1** | **-0.10** | -23.6 | -22.4 | 2.2 | 0.56 |
| Fixed wide (10bps) | -68.6 | -0.19 | -33.3 | -116.6 | 17.5 | 2.06 |
| Fixed mid (5bps) | -120.9 | -0.24 | -38.0 | -205.5 | 66.0 | 3.18 |
| Random | -347.5 | -0.80 | -80.3 | -591.0 | 103.6 | 4.33 |
| Fixed tight (0.5bps) | -678.6 | -0.91 | -141.5 | -1153.9 | 301.5 | 4.79 |
| A-S (calibrated) | -713.1 | -1.65 | -757.4 | -1212.7 | 278.0 | 0.94 |

**Findings**
1. C-PPO beats every baseline on all risk-adjusted metrics — qualitatively
   consistent with the paper.
2. However, ALL strategies lose money. Under bar-based fill-at-touch
   simulation, passive fills are systematically adversely selected: a
   fill happens exactly when price trades through the quote, and the
   bar's closing mid tends to be on the wrong side. The paper's
   event-level simulator does not have this artifact to the same degree.
3. The agent's "win" is mostly learned abstention (2.2 fills/ep vs 300
   for tight quoting): it discovered fills are toxic in this simulator
   and quotes wide. Economically sensible given (2), but it means the
   current setup rewards avoidance rather than market making.
4. Next: tick-based side-aware fills (bid fills only on seller-initiated
   prints) should reduce the adverse-selection artifact and make the
   comparison meaningful. Then re-run and compare.

---

## Run 2 — C-PPO vs benchmarks, tick-based side-aware fills (2026-07-23)

Same setup as Run 1 except fills: a bid fills only when a
seller-initiated trade prints at/below it, an ask only when a
buyer-initiated trade prints at/above it (derived from the 69M-row
tick tape). Training: 500 updates, 838s wall-clock.

| Strategy | PnL/ep | Sharpe | PnLMAP | ND-PnL | Fills/ep | Mean inv |
|---|---|---|---|---|---|---|
| Fixed tight (0.5bps) | +250.8 | 0.57 | 55.2 | 426.5 | 543.7 | 4.54 |
| Fixed mid (5bps) | +175.3 | 0.33 | 46.7 | 298.1 | 89.8 | 3.75 |
| Fixed wide (10bps) | +99.9 | 0.28 | 50.6 | 169.8 | 22.0 | 1.97 |
| A-S (calibrated) | +94.1 | **0.99** | **189.8** | 160.1 | 446.9 | **0.50** |
| C-PPO (trained) | +36.5 | 0.24 | 47.1 | 62.0 | 3.5 | 0.78 |
| Random | -157.2 | -0.37 | -34.4 | -267.3 | 137.7 | 4.57 |

**Findings**
1. Under realistic side-aware fills, market making IS profitable —
   every non-random strategy flips to positive PnL. This confirms
   Run 1's losses were an artifact of the fill-at-touch model, not a
   property of the market.
2. A-S is now the best risk-adjusted strategy (Sharpe 0.99, PnLMAP
   189.8) with the smallest inventory (0.50): its inventory-skewing
   mechanism works exactly as the theory promises. A strong baseline,
   consistent with the paper's finding that "AS is good at inventory
   controlling".
3. Our lightweight C-PPO is profitable but under-trades (3.5 fills/ep)
   — it carried over the conservative style learned under a hostile
   reward landscape and did not discover aggressive spread capture in
   500 updates. In the paper, C-PPO's edge came with Attn-LOB
   pre-training; their own ablation shows performance collapses
   without learned LOB representations. Our result is consistent:
   without pre-training, RL does not beat the classical formula.
4. Next steps, in order of expected value: (a) mid-price direction
   pre-training of the encoder (tests H1/H4 directly), (b) longer
   training / entropy schedule so the agent explores tighter quoting,
   (c) volume-aware fills (queue position) as a further realism step.

**Fill-model sensitivity (Run 1 vs Run 2, same strategies)**

| Strategy | PnL/ep (snapshot fills) | PnL/ep (tick fills) |
|---|---|---|
| Fixed tight | -678.6 | +250.8 |
| A-S | -713.1 | +94.1 |
| C-PPO | -13.1 | +36.5 |

The fill assumption alone swings results by ~900 USD/episode —
methodologically, simulator fidelity dominates strategy choice at
this data frequency.

---

## Attribution follow-up — what actually caused the Run 1 → Run 2 flip (2026-07-30)

Run 1 → Run 2 changed two things at once (fill-price source AND side
filter) — a confound. Decisive test: same benchmarks, same 73 episodes,
three fill variants:

| Fill variant | Fixed tight | A-S | Random |
|---|---|---|---|
| A: precomputed min/max cols, no side filter (=Run 1) | -678.6 | -713.1 | -347.5 |
| B: true tick tape, no side filter | +257.6 | +98.2 | -201.9 |
| C: true tick tape, side-aware (=Run 2) | +250.8 | +94.1 | -180.9 |

**The flip is A→B (data source), not B→C (side filter, ~2% effect).**
The precomputed min/max_trade_price columns leave "phantom fill" room
(price below the interval's true low) in 17.7% of intervals, median
0.36 bps — the same order as a tight half-spread, so tight quoting
gets systematically filled at prices that never printed. Corrected
claim: simulator fidelity dominates via INPUT DATA INTEGRITY ($900/ep);
side-awareness is a small correctness refinement. Confirming the
precomputed columns' definition with Brian is now a priority question.

### External validation vs Binance official klines (2026-07-30)

Sample window 2022-10-20 00:00-01:00 UTC, 60 one-minute klines from
the Binance REST API as independent ground truth:

| Source | Low exact | High exact | Trades/min |
|---|---|---|---|
| Binance klines (referee) | — | — | 4,699 |
| Our ticks parquet | 100% (median dev $0.00) | 100% | 4,699 (exact) |
| Precomputed trade columns | 5% (median dev $1.95) | 2% | 436 (~9%) |

The ticks parquet IS the full Binance BTCUSDT tape; the precomputed
min/max/count columns trace to a ~10x sparser source with ~1bp median
range deviation — the scale that drove Run 1's phantom fills. Rounding
ruled out (tolerance sweep). Question for Brian: what feed/sampling
produced these columns?

### Finding 3 — quote-information timing dominates at 10s cadence (2026-07-30)

Concrete case: LOB row 58953 (label 2022-10-27 06:49:20 UTC) carries
mid 20,714.84 / min_trade 20,713.90 — but the tape only reaches those
prices in the NEXT interval (06:49:20-30, crash to 20,709). Snapshot
content can lead its own label on fast intervals (collection lag:
median 0.9s, max 3.6s on the calm 10/20 rds sample; larger in fast
markets). Robustness bracket on the 73 test episodes:

| Quote basis | Fixed tight | A-S |
|---|---|---|
| Row-t mid (Run 2 as-run, zero-latency-or-better) | +250.8 (SR 0.57) | +94.1 (SR 0.99) |
| Previous-row mid (10s-stale, guaranteed no peek) | -622.8 (SR -0.90) | -647.6 (SR -1.56) |

Interpretation: absolute profitability at 10s cadence is fragile —
bracketed by information timing; relative comparisons under a fixed
regime remain meaningful. This is partly genuine latency sensitivity
(stale quoting at tight spreads loses regardless of data quality) and
partly snapshot clock provenance, which must be confirmed with Brian:
(1) what feed produced the trade-stat columns; (2) which clock stamps
the snapshot rows (exchange time vs database_time vs schedule label).

### Root cause identified — trade-stat columns cover (t, t+lag], not (t-10s, t] (2026-07-31)

Decisive test: taking each row's claimed trade_count and grabbing the
first `count` trades AFTER the row's label reproduces the claimed
min AND max exactly in 34% of rows; the implied window span has
median 0.96s — matching the rds database_time-minus-time lag (median
0.94s). Neighboring-window check shows claimed max often equals the
NEXT interval's tape max exactly. Conclusion: the columns were
computed over the collection-lag window after the schedule label
(~1s, ~10% of trades), not the trailing 10s interval. Run 1 therefore
judged fills against a sparse, forward-shifted price sample.
Question for Brian is now concrete: confirm the computation window.

---

- **Config drift bug (found while building Exp 2):** `TrainConfig.patience` had silently drifted to `10` in code while its own docstring, every notebook, and this file's Exp 1/1b writeups all said `5`. Nothing caught it until an unrelated rename-only re-run of notebook 05 produced quietly different numbers than what was documented. Root cause unclear (no single edit is implicated), but the practical lesson: a mismatch between a docstring and its own default value is exactly the kind of thing that survives silently unless something asserts on it. Fixed to `5` and pinned with a regression test (`test_default_patience_matches_documented_value`); Exp 1 and Exp 1b were both re-run under the corrected value and every number in this file reflects that re-run, not the original one. The qualitative findings (dropout delays overfitting, tuning buys a modest but real improvement, most of the gap to the paper is structural) held up across the fix — only the exact numbers and, notably, Exp 1b's *winning* candidate changed (`lower_lr_light` → `dropout_0.3`).
- Exp 1b: regularization's effect was entirely on *how much better* the best val_loss got, not primarily on how many epochs it took to get there (8 vs. 9 epochs across every candidate, barely different) — a different pattern than what the pre-fix run suggested (dropout mainly buying more epochs). Worth re-examining if patience is tuned differently in a future search.

---

## Experiment 4 — independent reproduction (2026-08-21)

The report's Exp 4 code was not in the repository, so we re-implemented
it from the report's Section 9 spec (src/paper_replication/rl/exp4.py,
scripts/run_exp4.py) and re-ran end to end: Attn-LOB pretrained fresh
with Exp 1b's winning config (41s on MPS; val_loss 0.9399 vs report's
0.9316; splits 35,467/8,867 match exactly), then C-PPO + D-DQN + five
baselines on the identical 2,364/591 episode grid (113s).

| Policy | Sharpe (report) | Sharpe (ours) |
|---|---|---|
| C-PPO | -0.788 | **-0.785** |
| D-DQN | -0.243 | **-0.094** |
| Random | -0.788 | -0.785 |
| Fixed 15% | -0.784 | -0.779 |
| Fixed 50% | -0.776 | -0.771 |
| Fixed 100% | -0.774 | -0.769 |
| AS | -0.775 | -0.772 |

All four headline findings reproduce independently: (1) every policy
loses; (2) D-DQN separates from the pack and beats C-PPO (H2
contradicted at this budget; D-DQN's exact number is seed-noisy, as in
the report's own re-run); (3) C-PPO is statistically indistinguishable
from random/fixed; (4) AS tracks Fixed(100%) — consistent with the
volatility-scaled spread pinning at the cap. PnLMAP magnitudes match
(~-30); D-DQN's PnLMAP is unstable in both runs (near-zero-inventory
denominator). Artifacts in results/exp4/.

### Exp 4 single-variable rerun — fills switched to tick side-aware (2026-08-27)

Identical to the Exp 4 reproduction in every respect (30-step episodes,
591 test episodes, frozen pretrained backbone, paper reward, USD caps,
same budgets and seed); ONLY the fill referee changes from
quote-through endpoints to tick-tape side-aware fills. Runtime 129s.

| Policy | Sharpe (quote-through) | Sharpe (tick fills) | PnL$/ep (tick) |
|---|---|---|---|
| C-PPO | -0.785 | **+0.201** | +0.0009 |
| D-DQN | -0.094 | **+1.207** | +0.0023 |
| Random | -0.785 | **+0.207** | +0.0009 |
| Fixed 15% | -0.779 | -0.044 | -0.0002 |
| Fixed 50% | -0.771 | **+0.198** | +0.0009 |
| Fixed 100% | -0.769 | **+0.385** | +0.0021 |
| AS | -0.772 | **+0.385** | +0.0021 |

Findings: (1) six of seven policies flip from negative to positive
Sharpe with everything else held fixed — the all-negative headline of
Exp 4 is a property of the endpoint fill model, not of the asset or
the strategies; (2) absolute levels remain economically tiny
(~$0.001/ep) because the $0.10 spread cap bounds earnings — the USD
cap transfer issue is a separate, additive distortion; (3) AS remains
exactly equal to Fixed(100%) under both referees, confirming the
gamma-unit degeneracy is referee-independent; (4) D-DQN leads under
both referees. Artifacts in results/exp4_tick/.

### Exp 4 monthly rerun — tick fills on 2022-10-10..11-10 (2026-09-02)

Same configuration and budgets as the tick-fill rerun above; only the
data window grows from 11 days (88,727 rows) to the full monthly file
(275,941 rows; 7,357 train / 1,839 test episodes). The monthly snapshot
file lacks trade columns and the tick tape only covers 10-19..10-30, so
per-interval side-aware extremes for the missing 20 days were built from
Binance official daily aggTrades (scripts/build_monthly_tick_extremes.py);
on the overlap day 2022-10-25 the two sources agree on 8,633/8,633
buy-side and 8,632/8,633 sell-side intervals (the one difference is the
midnight-boundary interval in the check itself). Interval coverage 100%.
Data prep 492s; training + eval 139s (vs 129s on 11 days — compute is
budget-bound, not data-bound). Frozen backbone reused from Exp 1b
(pretrained on 10-19..10-30, inside the monthly train split, disjoint
from the monthly test window).

| Policy | Sharpe (11d test, tick) | Sharpe (monthly test, tick) | PnL$/ep (monthly) |
|---|---|---|---|
| C-PPO | +0.201 | -0.191 | -0.0083 |
| D-DQN | +1.207 | -0.027 | -0.0000 |
| Random | +0.207 | -0.181 | -0.0080 |
| Fixed 15% | -0.044 | -0.202 | -0.0088 |
| Fixed 50% | +0.198 | -0.189 | -0.0084 |
| Fixed 100% | +0.385 | -0.167 | -0.0075 |
| AS | +0.385 | -0.168 | -0.0076 |

Findings: (1) this is NOT a single-variable comparison against the
11-day rerun — both the training data and the test period change. The
monthly 80/20 split places the test window on 11-04..11-10, which
contains the FTX collapse (mid 20,756 -> low 15,596, -15.2% over the
window; the 11-day test window was flat at -0.1%). (2) Under that
regime every policy is slightly negative under tick fills too: the
sign flip observed on the October window is period-dependent — the
honest statement is "the fill model decides the sign on a calm window;
a -15% crash week overwhelms spread capture at these caps regardless
of referee." (3) The relative ordering is preserved: D-DQN again
learns near-zero inventory and sits closest to flat (-0.027), and AS
still tracks Fixed(100%) (gamma degeneracy unchanged). (4) PnLMAP std
blows up (~108 vs ~17) — crash-week inventory marks dominate the
denominator. Artifacts in results/exp4_month_tick/.

### Exp 4 clean-window suite — expand the period, exclude the black swan (2026-09-03)

Daily-RV scan of the monthly file: 10-10..11-07 stays in 0.7-3.7%/day
(the CoinDesk story 11-02, the CZ tweet 11-06 and the crash eve 11-07
are all inside October's normal range; October's own max was CPI day
10-13 at 3.74%), then 11-08..11-10 jumps to 8.0-8.6% with 15-20%
intraday ranges. The regime break is exactly the three FTX days, so the
maximal clean window is 10-10..11-07 (29 days, 2.6x the 11-day file).
Four runs, all tick fills on the monthly file's grid
(scripts/run_exp4_clean_window.py):

| Run | Train | Test | Sharpe: C-PPO / D-DQN / Fixed100 / AS | Time |
|---|---|---|---|---|
| run1_clean29 | 10-10..11-02 (80%) | 11-02..11-07 | -0.25 / +0.02 / -0.18 / -0.18 | 140s |
| run1_clean29_budget (2.6x budget) | same | same | -0.25 / **-0.61** / -0.18 / -0.18 | 333s |
| run2a_fixedtest_9d | 10-19..10-28 | 10-28 18:37..10-30 | -0.27 / +0.09 / -0.19 / -0.19 | 127s |
| run2b_fixedtest_18d (1.9x data) | 10-10..10-28 | same | -0.28 / +0.04 / -0.19 / -0.19 | 136s |

Findings. (1) More training data changes nothing (run2a vs run2b is
single-variable: identical test episodes, 1.9x train data, all deltas
within noise) — at this budget C-PPO is statistically random and D-DQN
finds "stay flat" either way. (2) Scaling the budget 2.6x with the data
makes D-DQN WORSE (-0.61): longer training pushes it off the flat local
optimum into active quoting that loses — at these caps the learnable
optimum really is "don't trade". (3) run2a repeats the 11-day tick
run's calendar windows on the monthly grid and the slightly-positive
result becomes slightly-negative. Diagnosed and ruled out: grid density
(median interval 10.0s vs 10.1s), data gaps (none >20s on either grid),
mid-price bias between files (0.00 where timestamps coincide). What
remains is interval phase: the two files are independent ~10s samplings,
so interval boundaries land differently. With the $0.10 spread cap a
full round trip earns ~$0.0001 (0.001 BTC x $0.10), ~$0.003 per
30-step episode, and the sign is a knife-edge between that and 10-second
adverse drift — sampling phase alone flips it. (4) Takeaway for the
report: under unit-transferred USD caps every calm-window tick-fill
result is economically zero (|ND-PnL| < 0.011) with a referee-dependent
sign; only the FTX regime produces a signal that survives referee
construction. The unit-corrected caps rerun remains the prerequisite
for any sign claim. Artifacts in results/exp4_run*.

### Exp 4 Step A' — corrected caps + the paper's implicit touch floor (2026-09-04)

The paper never states a min-spread hyperparameter (Eq. 10 allows
spread=0), but its Fixed baselines quote at LOB levels 1-3 (never inside
the touch) and the A-share tick (~= the whole market spread) leaves no
room inside anyway. BTC's relatively ~1000x finer tick opens that region
up — and the capfix C-PPO promptly fell into it (learned $0.35 spread
inside the $0.54 touch, 55 fills/ep, -0.54 Sharpe). Mirror: clamp quotes
to the current best bid/ask (TOUCH_FLOOR in exp4.py). Caps stay
$2.70/$5.40; zeta still 0.01 (Step B pending).

| Sharpe | capfix 10d | capfloor 10d | capfix 29d | capfloor 29d |
|---|---|---|---|---|
| C-PPO | -0.537 | **+0.183** | -0.600 | -0.001 |
| D-DQN | 0.005 | 0.030 | 0.019 | 0.017 |
| Random | -0.407 | -0.062 | -0.297 | 0.005 |
| Fixed(15%) | 0.072 | 0.034 | 0.022 | 0.009 |
| AS | 0.012 | -0.010 | 0.004 | 0.003 |

Findings: (1) walling off the inside-the-touch region flips C-PPO from
worst-in-table to best-in-table on the October window — the -0.54 was a
toxic-region local optimum, not a PPO failure; Random improves for the
same reason. (2) On 1-hour-equivalent terms (x sqrt(12) aggregation)
capfloor C-PPO ~ 0.63, in the simple line's range — the official line
stops being "crushed" once the same scale mirrors are applied, and its
edge needs no signed-bias loophole. (3) The quiet 29d test window
(11-02..11-07) yields ~0 for every policy — regime-dependence persists.
Runtimes 127s/142s. Artifacts in results/exp4_capfloor_*.
