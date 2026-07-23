# Experiments

## Running Experiments

```bash
python -m paper_replication.experiments.exp1
```

## Results

| Experiment | Metric | Paper Result | Our Result | Delta |
|------------|--------|-------------|-----------|-------|
| Exp 1 | — | — | — | — |

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
