"""Market-making RL environment and networks (Guo et al., 2023, Section III-C/D/E)."""

from paper_replication.rl.action_space import (
    ContinuousActionSpace,
    DiscreteActionSpace,
    Quote,
)
from paper_replication.rl.baselines import (
    AvellanedaStoikovPolicy,
    FixedSpreadPolicy,
    RandomPolicy,
)
from paper_replication.rl.config import RLConfig
from paper_replication.rl.dqn import DQNConfig, train_dqn
from paper_replication.rl.env import MarketMakingEnv, Observation, StepResult
from paper_replication.rl.evaluate import evaluate_policy, run_episode
from paper_replication.rl.feature_cache import precompute_lob_features
from paper_replication.rl.market_data import (
    MarketDataset,
    build_market_dataset,
    chronological_split,
)
from paper_replication.rl.metrics import AggregateMetrics, EpisodeLog, aggregate_metrics
from paper_replication.rl.networks import (
    ContinuousActorCritic,
    DuelingQNetwork,
    RLFeatureExtractor,
    RLNetworkConfig,
)
from paper_replication.rl.ppo import PPOConfig, train_ppo
from paper_replication.rl.reward import (
    dampened_pnl,
    delta_pnl,
    hybrid_reward,
    inventory_punishment,
    trading_pnl,
)
from paper_replication.rl.simulator import Fill, MarketSimulator
from paper_replication.rl.state import agent_state_vector

__all__ = [
    "ContinuousActionSpace",
    "DiscreteActionSpace",
    "Quote",
    "AvellanedaStoikovPolicy",
    "FixedSpreadPolicy",
    "RandomPolicy",
    "RLConfig",
    "DQNConfig",
    "train_dqn",
    "MarketMakingEnv",
    "Observation",
    "StepResult",
    "evaluate_policy",
    "run_episode",
    "precompute_lob_features",
    "MarketDataset",
    "build_market_dataset",
    "chronological_split",
    "AggregateMetrics",
    "EpisodeLog",
    "aggregate_metrics",
    "ContinuousActorCritic",
    "DuelingQNetwork",
    "RLFeatureExtractor",
    "RLNetworkConfig",
    "PPOConfig",
    "train_ppo",
    "dampened_pnl",
    "delta_pnl",
    "hybrid_reward",
    "inventory_punishment",
    "trading_pnl",
    "Fill",
    "MarketSimulator",
    "agent_state_vector",
]
