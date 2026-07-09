# Paper Overview

## Citation

> Guo, H., Lin, J., & Huang, F. (2023). Market Making with Deep Reinforcement Learning from Limit Order Books. arXiv preprint arXiv:2305.15821.

## Summary

Guo, Lin, and Huang, in Market Making with Deep Reinforcement Learning from Limit Order Books, argue that reinforcement learning agents can achieve better market-making performance by learning market representations directly from raw limit order book (LOB) data rather than relying on handcrafted features or strong modeling assumptions. They propose a CNN-attention architecture (Attn-LOB) to automatically extract informative features from the LOB. They further introduce a continuous action space that enables finer quote placement and a hybrid reward function that balances profitability with inventory risk. Through experiments on three equities, they demonstrate that the resulting PPO-based agent outperforms discrete-action RL methods and classical market-making baselines. Overall, the paper shows that representation learning from raw LOB data can substantially improve reinforcement learning for market making.

The primary contribution of the paper is the proposal of Attn-LOB, a CNN-attention architecture that automatically learns informative market representations from raw limit order book data. Previous reinforcement learning approaches relied on handcrafted features, which may fail to capture the complex spatial and temporal relationships within the order book. The authors preprocess the LOB into stationary relative-price features and use convolutional layers, an Inception module, and multi-head self-attention to learn representations before reinforcement learning. Their ablation experiments demonstrate that removing the LOB representation or replacing Attn-LOB with a simple MLP leads to a substantial performance degradation, indicating that learned LOB representations are a key driver of the model's success. Therefore, the paper provides strong evidence that automatic feature extraction is more effective than handcrafted market representations.

The authors argue that market-making decisions are naturally continuous rather than discrete. Earlier reinforcement learning methods restricted the agent to a small number of predefined quoting actions, limiting its ability to adjust bid and ask prices. To overcome this limitation, the paper defines two continuous actions controlling reservation price bias and quoted spread, allowing the agent to generate more precise quotes. PPO is used because it naturally supports continuous action spaces, whereas Dueling DQN is limited to discrete actions. Experimental results show that the continuous-action PPO agent consistently outperforms the discrete-action DQN agent, demonstrating the practical value of continuous quote optimization.

The paper introduces a hybrid reward function that combines trading profit, dampened holding PnL, and an inventory penalty to better align the agent's objectives with those of a market maker. Using trading PnL alone can encourage speculative behaviour, as the agent may profit from directional price movements rather than earning the bid-ask spread, while holding large inventory positions exposes the agent to significant losses during adverse market movements. By incorporating dampened holding PnL and explicitly penalizing large inventories, the reward function encourages the agent to prioritize spread capture while effectively managing inventory risk. The experimental results demonstrate that this reward function enables the agent to achieve high profitability while maintaining relatively low inventory levels.

The paper demonstrates that the continuous-action PPO agent (C-PPO) outperforms the discrete-action Dueling DQN (D-DQN) and five baseline market-making strategies. This claim is supported by experimental results showing that C-PPO consistently achieves the highest PnLMAP values, indicating that it generates high trading profits while maintaining relatively low inventory levels. Although the classical Avellaneda-Stoikov (AS) strategy also exhibits effective inventory control, it does so at the expense of lower profitability. Furthermore, the latency experiments show that the C-PPO agent adapts its quoting behaviour to changing market conditions, suggesting that it has learned a dynamic market-making strategy rather than following fixed quoting rules.
 
The attention visualization demonstrates that the agent adapts its focus to different market conditions by varying the length of its attention window. During stable market conditions, the model places most of its attention on recent events, suggesting that the latest market information is the most informative for decision making. In contrast, during periods of rapid market change, the model attends to both recent and earlier events, indicating that a longer history is required to understand the evolving market dynamics. These results suggest that the attention mechanism enables the agent to adapt its decision-making process according to the current market regime rather than relying on a fixed historical window. These findings suggest that the agent exhibits adaptive behaviour similar to that of an experienced market maker, although the attention visualization alone does not establish that it has learned human-like decision making.

## Conclusion
Overall, this paper demonstrates that automatically learning market representations from raw limit order book data can significantly improve reinforcement learning-based market making compared to approaches that rely on handcrafted features or strong modeling assumptions. By combining the Attn-LOB architecture with a continuous action space and a hybrid reward function, the proposed framework achieves higher profitability while effectively controlling inventory risk. The experimental and ablation results provide strong evidence that learned LOB representations are the primary driver of the model's performance, while the continuous action space further improves quote optimization. This work is particularly relevant to my research interests in market making, as it highlights the importance of representation learning, reinforcement learning, and feature engineering for developing adaptive quoting strategies that better reflect real market dynamics.

## Limitations
Although the proposed framework demonstrates strong performance, several limitations remain. First, the experiments are conducted on only three Chinese equities, making it difficult to assess the model's generalizability across different markets and asset classes. Second, the evaluation is performed in a historical simulation that does not fully capture real-world market impact, latency, or competition with other market participants. Finally, the paper evaluates the model over relatively short trading episodes of approximately three to five minutes, leaving its long-term performance and robustness under different market regimes largely unexplored.

## Key Claims to Replicate

1. **Claim 1** — Description
2. **Claim 2** — Description

## Datasets

| Dataset | Source | Notes |
|---------|--------|-------|
| — | — | — |

## Deviations from Original

Document any intentional or forced differences here as the replication progresses.
