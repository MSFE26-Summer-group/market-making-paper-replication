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
