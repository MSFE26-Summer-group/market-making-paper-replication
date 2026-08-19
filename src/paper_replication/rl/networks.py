"""Networks for the RL phase (paper Fig. 1's "Concatenate" -> "Action Space MLP").

Reuses `AttnLOB.forward_features` as the LOB-state feature extractor (the
same backbone from Section III-B, optionally loaded from a pretrained
checkpoint), concatenates the paper's dynamic and agent state onto its
pooled output, and feeds the result through a small MLP trunk before
branching into task-specific heads:

- `ContinuousActorCritic` -- the PPO policy/value network for the continuous
  action space (paper III-C2, III-E): a Beta-distribution policy head over
  `(A1, A2) in [0, 1]^2` (Beta's support matches the action space exactly,
  unlike a Gaussian, which would need clipping/squashing not mentioned in
  the paper) plus a state-value head.
- `DuelingQNetwork` -- the Dueling DQN Q-network for the discrete action
  space (paper III-C1, III-E): value + advantage streams combined per Wang
  et al.'s dueling architecture, over the 8 discrete actions.

Both heads also expose a `forward_from_lob_features` path that skips the
Attn-LOB backbone and takes its pooled output directly. RL training keeps
the backbone frozen at its pretrained weights (paper Fig. 1's own
pretrain-then-RL structure) and its output doesn't depend on the agent's
actions -- only `agent_state`'s inventory feature does -- so
`rl.feature_cache.precompute_lob_features` computes it once, in large
batches, over an entire dataset up front. Every RL training step then does
one array lookup instead of one CNN+attention forward pass, which is what
makes training tractable in this replication's compute budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from paper_replication.models.attn_lob import AttnLOB, AttnLOBConfig


@dataclass(frozen=True)
class RLNetworkConfig:
    """Hyperparameters shared by both RL network heads.

    attn_lob_config: architecture of the LOB-state feature extractor. Pass a
        config matching a pretrained checkpoint's if loading one (see
        `training.checkpoint.load_checkpoint`) -- the RL networks only call
        `forward_features`, never the pretrain classification head.
    n_dynamic_features: width of the dynamic-state vector this dataset
        produces (`len(MarketDataset.dynamic_feature_names)`).
    n_agent_features: width of the agent-state vector (`state.N_AGENT_FEATURES`).
    trunk_hidden_dim: width of the shared MLP trunk after concatenation.
    """

    attn_lob_config: AttnLOBConfig = AttnLOBConfig()
    n_dynamic_features: int = 24
    n_agent_features: int = 2
    trunk_hidden_dim: int = 128

    @property
    def concat_dim(self) -> int:
        return (
            self.attn_lob_config.embed_dim
            + self.n_dynamic_features
            + self.n_agent_features
        )


class RLFeatureExtractor(nn.Module):
    """Shared trunk: Attn-LOB backbone + dynamic/agent state -> MLP (Fig. 1)."""

    def __init__(self, config: RLNetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.attn_lob = AttnLOB(config.attn_lob_config)
        self.trunk = nn.Sequential(
            nn.Linear(config.concat_dim, config.trunk_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.trunk_hidden_dim, config.trunk_hidden_dim),
            nn.LeakyReLU(),
        )

    def forward(
        self,
        lob_state: torch.Tensor,
        dynamic_state: torch.Tensor,
        agent_state: torch.Tensor,
    ) -> torch.Tensor:
        """(batch,T,4n), (batch,n_dyn), (batch,n_agent) -> (batch, trunk_hidden_dim)."""
        lob_features = self.attn_lob.forward_features(lob_state)
        return self.forward_from_lob_features(lob_features, dynamic_state, agent_state)

    def forward_from_lob_features(
        self,
        lob_features: torch.Tensor,
        dynamic_state: torch.Tensor,
        agent_state: torch.Tensor,
    ) -> torch.Tensor:
        """Same as `forward`, but starting from an already-pooled `(batch, embed_dim)`
        Attn-LOB output (see module docstring) instead of a raw `lob_state` window.
        """
        combined = torch.cat([lob_features, dynamic_state, agent_state], dim=-1)
        trunk_out: torch.Tensor = self.trunk(combined)
        return trunk_out


class ContinuousActorCritic(nn.Module):
    """PPO policy/value network for the continuous action space (paper III-C2, III-E).

    Policy head outputs `(alpha, beta)` Beta-distribution parameters for
    each of A1, A2. `softplus(...) + 1` keeps both shape parameters > 1, so
    the resulting Beta density is always unimodal (never U-shaped) --
    a reasonable default for a market-making quote policy, since the paper
    doesn't specify a distribution family.
    """

    def __init__(self, config: RLNetworkConfig) -> None:
        super().__init__()
        self.extractor = RLFeatureExtractor(config)
        self.policy_head = nn.Linear(config.trunk_hidden_dim, 4)  # (alpha, beta) x 2
        self.value_head = nn.Linear(config.trunk_hidden_dim, 1)

    def forward(
        self,
        lob_state: torch.Tensor,
        dynamic_state: torch.Tensor,
        agent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (alpha (batch, 2), beta (batch, 2), value (batch,))."""
        features = self.extractor(lob_state, dynamic_state, agent_state)
        return self._heads(features)

    def forward_from_lob_features(
        self,
        lob_features: torch.Tensor,
        dynamic_state: torch.Tensor,
        agent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as `forward`, from a precomputed Attn-LOB output.

        See module docstring.
        """
        features = self.extractor.forward_from_lob_features(
            lob_features, dynamic_state, agent_state
        )
        return self._heads(features)

    def _heads(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.policy_head(features)
        alpha = nn.functional.softplus(raw[:, :2]) + 1.0
        beta = nn.functional.softplus(raw[:, 2:]) + 1.0
        value = self.value_head(features).squeeze(-1)
        return alpha, beta, value


class DuelingQNetwork(nn.Module):
    """Dueling DQN Q-network for the discrete action space (paper III-C1, III-E).

    Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)) -- Wang et al.'s dueling
    decomposition, matching the paper's own choice of Dueling DQN as the
    discrete-space training algorithm.
    """

    def __init__(self, config: RLNetworkConfig, n_actions: int = 8) -> None:
        super().__init__()
        self.extractor = RLFeatureExtractor(config)
        self.value_head = nn.Linear(config.trunk_hidden_dim, 1)
        self.advantage_head = nn.Linear(config.trunk_hidden_dim, n_actions)

    def forward(
        self,
        lob_state: torch.Tensor,
        dynamic_state: torch.Tensor,
        agent_state: torch.Tensor,
    ) -> torch.Tensor:
        """-> (batch, n_actions) Q-values."""
        features = self.extractor(lob_state, dynamic_state, agent_state)
        return self._q_values(features)

    def forward_from_lob_features(
        self,
        lob_features: torch.Tensor,
        dynamic_state: torch.Tensor,
        agent_state: torch.Tensor,
    ) -> torch.Tensor:
        """Same as `forward`, from a precomputed Attn-LOB output.

        See module docstring.
        """
        features = self.extractor.forward_from_lob_features(
            lob_features, dynamic_state, agent_state
        )
        return self._q_values(features)

    def _q_values(self, features: torch.Tensor) -> torch.Tensor:
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        centered_advantage = advantage - advantage.mean(dim=-1, keepdim=True)
        q_values: torch.Tensor = value + centered_advantage
        return q_values
