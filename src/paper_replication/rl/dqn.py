"""Dueling DQN training for `DuelingQNetwork` over `MarketMakingEnv` (paper III-E).

Standard DQN with a target network and experience replay (Mnih et al.,
2015), combined with the dueling architecture (`networks.DuelingQNetwork`,
Wang et al.) -- the paper names Dueling DQN as its discrete-action-space
training algorithm but doesn't specify its own hyperparameters, so these are
standard defaults, not values taken from the paper.

Like `ppo.py`, trains only the trunk + value/advantage heads against a
precomputed, frozen Attn-LOB feature cache (`feature_cache.py`).
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from paper_replication.rl.env import MarketMakingEnv
from paper_replication.rl.networks import DuelingQNetwork

Float32Array: TypeAlias = npt.NDArray[np.float32]


@dataclass(frozen=True)
class DQNConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    buffer_size: int = 20000
    batch_size: int = 64
    warmup_steps: int = 500
    train_interval: int = 4
    target_update_interval: int = 200
    log_interval: int = 200
    n_steps: int = 10000
    max_grad_norm: float = 5.0
    device: str = "cpu"
    seed: int = 0


@dataclass
class _Transition:
    lob_features: Float32Array
    dynamic_state: Float32Array
    agent_state: Float32Array
    action: int
    reward: float
    next_lob_features: Float32Array
    next_dynamic_state: Float32Array
    next_agent_state: Float32Array
    done: bool


class _ReplayBuffer:
    """Fixed-capacity ring buffer with O(1) push and O(batch_size) sampling."""

    def __init__(self, capacity: int, seed: int) -> None:
        self._capacity = capacity
        self._data: list[_Transition] = []
        self._pos = 0
        self._rng = random.Random(seed)

    def push(self, transition: _Transition) -> None:
        if len(self._data) < self._capacity:
            self._data.append(transition)
        else:
            self._data[self._pos] = transition
            self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int) -> list[_Transition]:
        indices = self._rng.sample(range(len(self._data)), batch_size)
        return [self._data[i] for i in indices]

    def __len__(self) -> int:
        return len(self._data)


def _epsilon_at(step: int, config: DQNConfig) -> float:
    frac = min(1.0, step / config.epsilon_decay_steps)
    return config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)


def _to_batch_tensor(values: list[Float32Array], device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack(values)).to(device)


def _update(
    model: DuelingQNetwork,
    target_model: DuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    batch: list[_Transition],
    config: DQNConfig,
    device: torch.device,
) -> float:
    lob = _to_batch_tensor([t.lob_features for t in batch], device)
    dyn = _to_batch_tensor([t.dynamic_state for t in batch], device)
    agent = _to_batch_tensor([t.agent_state for t in batch], device)
    next_lob = _to_batch_tensor([t.next_lob_features for t in batch], device)
    next_dyn = _to_batch_tensor([t.next_dynamic_state for t in batch], device)
    next_agent = _to_batch_tensor([t.next_agent_state for t in batch], device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
    rewards = torch.tensor(
        [t.reward for t in batch], dtype=torch.float32, device=device
    )
    dones = torch.tensor(
        [float(t.done) for t in batch], dtype=torch.float32, device=device
    )

    q_values = model.forward_from_lob_features(lob, dyn, agent)
    q_taken = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_q = target_model.forward_from_lob_features(next_lob, next_dyn, next_agent)
        next_q_max = next_q.max(dim=-1).values
        target = rewards + config.gamma * (1.0 - dones) * next_q_max

    loss = nn.functional.mse_loss(q_taken, target)
    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()
    return float(loss.item())


def train_dqn(
    env: MarketMakingEnv,
    model: DuelingQNetwork,
    lob_feature_cache: Float32Array,
    config: DQNConfig,
    train_episode_indices: list[int] | None = None,
) -> list[dict[str, float]]:
    """Trains `model` in place over `env`'s episodes; returns metrics logged every
    `config.log_interval` steps.

    `train_episode_indices`: episode indices to reset into (defaults to all
    of `env`'s episodes) -- pass the training portion of a
    `market_data.chronological_split` to keep held-out test episodes untouched.
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    model.to(device)

    n_actions = model.advantage_head.out_features
    target_model = copy.deepcopy(model).to(device)
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    buffer = _ReplayBuffer(config.buffer_size, config.seed)

    indices = (
        list(range(env.n_episodes))
        if train_episode_indices is None
        else list(train_episode_indices)
    )

    history: list[dict[str, float]] = []
    losses: list[float] = []
    episode_rewards: list[float] = []
    episode_reward = 0.0

    obs = env.reset(episode_index=int(rng.choice(indices)))
    for step in range(1, config.n_steps + 1):
        row = env.current_row
        lob_feat = lob_feature_cache[row]
        dyn = obs.dynamic_state.astype(np.float32)
        agent = obs.agent_state.astype(np.float32)

        epsilon = _epsilon_at(step, config)
        if rng.random() < epsilon:
            action = int(rng.integers(0, n_actions))
        else:
            model.eval()
            with torch.no_grad():
                lob_t = torch.from_numpy(lob_feat).unsqueeze(0).to(device)
                dyn_t = torch.from_numpy(dyn).unsqueeze(0).to(device)
                agent_t = torch.from_numpy(agent).unsqueeze(0).to(device)
                q_values = model.forward_from_lob_features(lob_t, dyn_t, agent_t)
                action = int(q_values.argmax(dim=-1).item())

        result = env.step(action)
        episode_reward += result.reward

        if result.done:
            next_lob_feat = np.zeros_like(lob_feat)
            next_dyn = np.zeros_like(dyn)
            next_agent = np.zeros_like(agent)
        else:
            assert result.observation is not None
            next_lob_feat = lob_feature_cache[env.current_row]
            next_dyn = result.observation.dynamic_state.astype(np.float32)
            next_agent = result.observation.agent_state.astype(np.float32)

        buffer.push(
            _Transition(
                lob_features=lob_feat,
                dynamic_state=dyn,
                agent_state=agent,
                action=action,
                reward=result.reward,
                next_lob_features=next_lob_feat,
                next_dynamic_state=next_dyn,
                next_agent_state=next_agent,
                done=result.done,
            )
        )

        if result.done:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            obs = env.reset(episode_index=int(rng.choice(indices)))
        else:
            assert result.observation is not None
            obs = result.observation

        if len(buffer) >= config.warmup_steps and step % config.train_interval == 0:
            model.train()
            batch = buffer.sample(config.batch_size)
            losses.append(
                _update(model, target_model, optimizer, batch, config, device)
            )

        if step % config.target_update_interval == 0:
            target_model.load_state_dict(model.state_dict())

        if step % config.log_interval == 0:
            history.append(
                {
                    "step": float(step),
                    "epsilon": epsilon,
                    "mean_recent_episode_reward": (
                        float(np.mean(episode_rewards[-10:]))
                        if episode_rewards
                        else float("nan")
                    ),
                    "mean_recent_loss": (
                        float(np.mean(losses[-50:])) if losses else float("nan")
                    ),
                }
            )

    return history
