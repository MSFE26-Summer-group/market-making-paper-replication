# Hypotheses

This document extracts the primary hypotheses from *Market Making with Deep Reinforcement Learning from Limit Order Books* (Guo et al., 2023). Each hypothesis is explicitly stated and paired with a validation strategy to guide the replication experiments.

---

## H1 — Automatic LOB Representation Learning Improves Market Making

### Paper Claim

Automatically learning market representations from raw Limit Order Book (LOB) data using Attn-LOB is more effective than relying on handcrafted market features.

### Subject

Market representation learning.

### Dependent Variables

- Normalized Daily PnL (ND-PnL)
- PnLMAP
- Sharpe Ratio

### Independent Variable

Feature representation.

- Attn-LOB (CNN + Inception + Self-Attention)
- Handcrafted market features

### Expected Outcome

The Attn-LOB representation should produce significantly higher profitability and risk-adjusted returns than handcrafted feature representations.

### Validation Method

Replicate the Attn-LOB model and compare its performance against reinforcement learning agents using handcrafted LOB features.

The hypothesis will be considered supported if Attn-LOB consistently achieves superior:

- ND-PnL
- PnLMAP
- Sharpe Ratio

---

## H2 — Continuous Action Spaces Improve Market Making Performance

### Paper Claim

A continuous action space trained using PPO produces better market-making performance than a discrete action space trained using Dueling DQN.

### Subject

Action space design.

### Dependent Variables

- ND-PnL
- PnLMAP
- Sharpe Ratio

### Independent Variable

Action space.

- Continuous
- Discrete

### Expected Outcome

The PPO agent operating in a continuous action space should outperform the Dueling DQN agent operating in a discrete action space across all evaluation metrics.

### Validation Method

Implement both action-space formulations and compare their reported metrics with those presented in the paper.

---

## H3 — Hybrid Reward Functions Improve Risk-Adjusted Performance

### Paper Claim

A hybrid reward function combining trading profit, dampened holding PnL, and inventory penalties produces better market-making behaviour than simpler reward formulations.

### Subject

Reward function design.

### Dependent Variables

- PnLMAP
- Inventory level
- Sharpe Ratio

### Independent Variable

Reward function.

- Hybrid reward
- Trading PnL only

### Expected Outcome

The hybrid reward should encourage spread capture while reducing speculative behaviour and excessive inventory accumulation.

### Validation Method

Train identical agents using both reward formulations and compare:

- Profitability
- Inventory exposure
- Risk-adjusted returns

---

## H4 — Learned LOB Representations Are the Primary Driver of Performance

### Paper Claim

The learned LOB representation contributes more to performance than other architectural components.

### Subject

Model architecture.

### Dependent Variables

- ND-PnL
- PnLMAP
- Sharpe Ratio

### Independent Variable

Model configuration.

- Full model
- Without LOB representation
- Attn-LOB replaced with MLP
- Without dynamic state

### Expected Outcome

Removing the LOB representation or replacing Attn-LOB with a simple MLP should significantly reduce performance, whereas removing the dynamic state should have only a minor effect.

### Validation Method

Replicate the ablation study and compare the performance degradation caused by removing each component.

---

## H5 — Attention Learns Adaptive Temporal Dependencies

### Paper Claim

The attention mechanism dynamically adjusts the historical information used for decision making according to market conditions.

### Subject

Temporal attention.

### Dependent Variables

- Attention weights
- Attention visualization

### Independent Variable

Market condition.

- Stable market
- Rapidly changing market

### Expected Outcome

During stable market conditions, the model should focus primarily on recent events.

During volatile market conditions, the model should attend to both recent and earlier events.

### Validation Method

Replicate the attention visualization presented in the paper and compare the attention distributions across different market regimes.

---

## Hypothesis Revision Log

This section documents any changes made to the hypotheses during the replication.

| Version | Hypothesis | Reason for Revision | Date |
|----------|------------|--------------------|------|
| v1.0 | Initial hypotheses extracted from the paper. | Initial documentation. | YYYY-MM-DD |

Future revisions should **never overwrite previous hypotheses**. Instead, append revisions to maintain a complete research record.

---

## Success Criteria

The replication will be considered successful if:

- The implementation reproduces the architecture described in the paper.
- The reproduced metrics follow the same relative ordering reported by the authors.
- The ablation study demonstrates similar trends.
- Continuous PPO outperforms discrete DQN.
- Attention visualizations exhibit similar adaptive behaviour.
- Any deviations from the original paper are documented and justified.
