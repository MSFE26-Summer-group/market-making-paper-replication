"""Attn-LOB encoder and PPO actor-critic (Guo, Lin & Huang 2023).

Architecture: 1-D convolutions over the time axis extract local LOB
patterns; multi-head self-attention weighs how much history matters;
the pooled embedding is concatenated with auxiliary features
(inventory, order imbalance) and fed to separate actor/critic heads.
The actor parameterizes a Gaussian over the 2-D continuous action
(reservation bias, spread).
"""

import torch
from torch import Tensor, nn
from torch.distributions import Normal


class AttnLOB(nn.Module):
    """CNN + self-attention encoder for (window, 40) LOB states."""

    def __init__(
        self, n_features: int = 40, d_model: int = 64, n_heads: int = 4
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, window, features) -> conv wants (batch, features, window)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm(h + a)
        pooled: Tensor = h.mean(dim=1)  # (batch, d_model)
        return pooled


class ActorCritic(nn.Module):
    """PPO policy and value heads on top of the Attn-LOB embedding."""

    def __init__(
        self,
        n_features: int = 40,
        n_aux: int = 3,
        d_model: int = 64,
        n_actions: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = AttnLOB(n_features, d_model)
        self.actor = nn.Sequential(
            nn.Linear(d_model + n_aux, 64), nn.Tanh(), nn.Linear(64, n_actions)
        )
        self.log_std = nn.Parameter(torch.full((n_actions,), -0.5))
        self.critic = nn.Sequential(
            nn.Linear(d_model + n_aux, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def _embed(self, state: Tensor, aux: Tensor) -> Tensor:
        return torch.cat([self.encoder(state), aux], dim=-1)

    def forward(self, state: Tensor, aux: Tensor) -> tuple[Normal, Tensor]:
        z = self._embed(state, aux)
        mean = torch.tanh(self.actor(z))  # keep actions in [-1, 1]
        dist = Normal(mean, self.log_std.exp())
        value = self.critic(z).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(self, state: Tensor, aux: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample an action; returns (action, log_prob, value)."""
        dist, value = self(state, aux)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), value
