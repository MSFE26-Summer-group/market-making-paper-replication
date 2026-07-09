# Paper Overview

## Citation

> Guo, H., Lin, J., & Huang, F. (2023). Market Making with Deep Reinforcement Learning from Limit Order Books. arXiv preprint arXiv:2305.15821.

## Summary

In "Market Making with Deep Reinforcement Learning from Limit Order Books," Guo, Lin, and Huang propose an end-to-end reinforcement learning framework for market making that automatically learns informative market representations directly from limit order book data using a CNN-attention network, overcoming the limitations of prior approaches that relied on handcrafted features and strong modeling assumptions.

The second contribution is the continuous action space, which is much more flexible than the discrete 8-action space as it allows much higher granularity of quotes for the agent to explore, and the optimal bid and ask price is obtained by calculating the reservation price, which is obtained using a bias value from Action A1 and the spread controlled by Action A2.

The paper designs a hybrid reward function combining trading profit, dampened holding PnL, and an inventory penalty, and shows that agents trained with it achieve strong profitability while maintaining low inventory risk relative to baseline strategies.

The paper shows that C-PPO, which uses the continuous action space, performs much better than the discrete-action-space algorithm D-DQN and the five baselines, especially on the PnLMAP metric, which depicts C-PPO attaining exceeding profits while maintaining low inventory levels.
 
The attention visualization shows how the agent adapts to different market conditions by paying most attention to near-past events while still looking at earlier changes, which depicts that the agent can learn market-making skills like humans.

This paper addresses an issue in past RL market-making agents through automatic feature extraction from LOB data compared to baselines; however, the generalizability can be questioned as it has only been tested on three assets.

## Key Claims to Replicate

1. **Claim 1** — Description
2. **Claim 2** — Description

## Datasets

| Dataset | Source | Notes |
|---------|--------|-------|
| — | — | — |

## Deviations from Original

Document any intentional or forced differences here as the replication progresses.
