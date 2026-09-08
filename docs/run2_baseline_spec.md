# Run 2 Baseline Specification — ablation row zero

Run 2 (tick-based side-aware fills, 500 PPO updates) is the frozen
baseline every future upgrade is compared against, on the identical
73 held-out episodes. This document records every difference vs
Guo, Lin & Huang (2023) so that no deviation is silent.

Status legend: ALIGNED / ADAPTED (justified deviation) / GAP (roadmap).

## Data
| Item | Paper | Ours | Status |
|---|---|---|---|
| Market | Shenzhen A-shares, 3 stocks | Binance BTC-USDT, 1 pair | ADAPTED (contribution) |
| Data type | event-by-event LOB | 10s snapshots + tick tape | ADAPTED (root of most items below) |
| Period | Nov 2019, 21 days, ~5M events | Oct 20-30 2022, 11 days, 88,678 snaps | ADAPTED |
| Sessions | stable intraday hours only | 24/7 | ADAPTED (incl. overnight regimes) |
| Split | 10d train (20% val) / 11d test | 70/30 chronological, no val set | ADAPTED (no validation set) |

## State
| Item | Paper | Ours | Status |
|---|---|---|---|
| Levels n / window T | 10 / 50 | 10 / 50 | ALIGNED |
| Price transform | ref[18] p/mid-1, then z-norm | same (bps), no 2nd z-norm | ALIGNED (minor) |
| Size normalization | max-norm | log1p + z-score (train stats) | ADAPTED (heavy tails) |
| Mid-change feature (ref[18]) | yes | missing | GAP |
| Dynamic state | 24 dims (OSI18+RV3+RSI3) | 1 dim (lob_imbalance) | GAP (largest state gap) |
| Agent state | inventory + remaining time | + imbalance | ALIGNED+ |

## Model
| Item | Paper | Ours | Status |
|---|---|---|---|
| Encoder | Conv2D + Inception + MHA | Conv1d x2 + MHA, d=64 | ADAPTED (lightweight) |
| Pre-training | mid-price direction task | none | GAP (H1/H4, top priority) |

## Action
| Item | Paper | Ours | Status |
|---|---|---|---|
| Continuous (bias, spread) | yes | yes | ALIGNED |
| Bias direction | Sign(inventory) forced | signed, learned | ADAPTED (more freedom) |
| Caps | 0.05 / 0.1 CNY | 5bps bias, 0.5-10bps half | ADAPTED (scale mapping) |
| Trade unit | 100 shares | 1 unit | ADAPTED (equivalent) |

## Reward
| Item | Paper | Ours | Status |
|---|---|---|---|
| eta / zeta / omega | 0.5 / 0.01 / 10 | same | ALIGNED |
| Formula | R = DP+TP-IP (DP dampens total) | TP + dampened(holding) - IP | ADAPTED (minor, same intent) |
| TP mid reference | at transaction time | bar-end mid | ADAPTED (bar data) |

## Simulator
| Item | Paper | Ours | Status |
|---|---|---|---|
| Decision frequency | per event (~0.1s) | per 10s | ADAPTED (data-driven) |
| Episode | 2000 events (3-5 min) | 360 steps (1 h) | ADAPTED (redesigned) |
| Fills | exact event replay | side-aware tick fills, <=1 unit/side/step, no queue/volume | ADAPTED (optimistic) |
| Fees / impact | 0 / ignored | same | ALIGNED (same simplification) |
| Episode-end liquidation | yes | yes | ALIGNED |
| Known data risk | - | snapshot clock ~1s lag (timing bracket measured) | ADAPTED (documented) |

## Training & evaluation
| Item | Paper | Ours | Status |
|---|---|---|---|
| C-PPO | yes | yes (hyperparams ours; paper silent) | ALIGNED |
| D-DQN | yes | missing | GAP (H2) |
| Metrics | ND-PnL, PnLMAP, PR, Sharpe | all but PR (fees=0) | ADAPTED |
| Baselines | AS, Random, Fixed1-3, Inv-RL, LOB-RL | AS (calibrated), Random, Fixed x3 | ADAPTED (2 legacy RL missing) |
| Latency experiment | yes | no | GAP (low priority) |
| Ablation + attention viz | yes | no | GAP (H5, after pre-training) |

## Tally & roadmap
ALIGNED 11 / ADAPTED 15 / GAP 6.
Gap priority: (1) pre-training [H1/H4] -> (2) dynamic state 3->27 dims
-> (3) D-DQN [H2] -> (4) mid-change feature -> (5) ablation & attention
viz [H5] -> (6) latency. Each upgrade: one variable, same 73 episodes,
report delta-Sharpe vs this baseline (C-PPO SR 0.24; A-S SR 0.99).
