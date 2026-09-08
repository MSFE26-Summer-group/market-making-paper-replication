# Pipeline Audit & Consolidated Results

Audit date: 2026-07-24 · branch `feature/rl-sample-model`

## Correctness checks (all programmatically verified)

| Check | Result |
|---|---|
| Look-ahead in states | PASS — `states[i]` ends at snapshot i by construction |
| Fill timing | PASS — quote uses `mid[t]`, fills use interval (t, t+1] only |
| Tick fill coverage | PASS — 0% intervals missing either side (BTC is dense) |
| Train/test separation | PASS — last train episode ends at 62,073 < first test start 62,074 |
| Test episode design | PASS — 73 non-overlapping 1h episodes, identical across policies |
| Normalization leakage | PASS — size stats from first 70% only |
| Test suite | PASS — 16 tests, 79% coverage, mypy clean |

## Paper alignment

| Item | Paper | Ours | Status |
|---|---|---|---|
| LOB levels n / window T | 10 / 50 | 10 / 50 | aligned |
| Price stationarity transform | ref [18]: p/p_mid − 1 | same (×1e4, bps) | aligned |
| Volume normalization | max-norm | log1p + z-score (train stats) | deviation (heavy-tailed crypto sizes) |
| Agent state | inventory, remaining time | + lob_imbalance | aligned+ |
| Action space | continuous (bias, spread), A1 magnitude with Sign(inv) | signed bias learned directly | deviation (more freedom) |
| eta / zeta / omega | 0.5 / 0.01 / 10 | 0.5 / 0.01 / 10 | aligned |
| Reward (eq 16) | R = DP + TP − IP, where DP dampens total ΔPnL and TP is added on top | TP + dampened(holding) − IP; trading PnL counted once, dampening applied to holding only | minor deviation, same intent — flagged for discussion |
| Fill simulation | event-by-event exact | side-aware tick fills (price crosses; no queue/volume) | approximation |
| Episode | 2000 events (3–5 min) | 360 steps (1 h) | redesigned (event clock ≠ wall clock; BTC ≈ 67 trades/s) |
| Pre-training (Attn-LOB) | mid-price direction task | not yet | pending — next milestone (H1/H4) |
| D-DQN baseline | yes | not yet | pending (H2) |

## Known data caveat

`min/max_trade_price` columns in the LOB parquet match tick-derived
per-interval extremes in only 10–21% of rows under any alignment —
their exact window/definition needs confirmation from Brian. Run 2
derives fills directly from the tick tape and is unaffected.

## Consolidated results (73 held-out 1h episodes, BTC-USDT)

Training: 500 PPO updates × 4 episodes/update.
Machine: Apple Silicon (CPU only).
Run 1 training time: 833 s. Run 2 training time: 838 s.

### Run 2 — tick-based side-aware fills (trusted run)

| Strategy | PnL/ep (USD/unit) | Sharpe | PnLMAP | ND-PnL | Fills/ep | Mean inv |
|---|---|---|---|---|---|---|
| Fixed tight (0.5 bps) | +250.8 | 0.57 | 55.2 | 426.5 | 543.7 | 4.54 |
| Fixed mid (5 bps) | +175.3 | 0.33 | 46.7 | 298.1 | 89.8 | 3.75 |
| Fixed wide (10 bps) | +99.9 | 0.28 | 50.6 | 169.8 | 22.0 | 1.97 |
| A-S (calibrated) | +94.1 | **0.99** | **189.8** | 160.1 | 446.9 | **0.50** |
| C-PPO | +36.5 | 0.24 | 47.1 | 62.0 | 3.5 | 0.78 |
| Random | −157.2 | −0.37 | −34.4 | −267.3 | 137.7 | 4.57 |

### Run 1 — snapshot fill-at-touch (kept for the methodology finding)

| Strategy | PnL/ep | Sharpe | Fills/ep |
|---|---|---|---|
| C-PPO | −13.1 | −0.10 | 2.2 |
| Fixed wide | −68.6 | −0.19 | 17.5 |
| Fixed mid | −120.9 | −0.24 | 66.0 |
| Random | −347.5 | −0.80 | 103.6 |
| Fixed tight | −678.6 | −0.91 | 301.5 |
| A-S | −713.1 | −1.65 | 278.0 |

Fill assumption alone swings per-episode PnL by ~900 USD — simulator
fidelity dominates strategy choice at this data frequency.
