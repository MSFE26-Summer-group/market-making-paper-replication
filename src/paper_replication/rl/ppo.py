"""PPO training for `ContinuousActorCritic` over `MarketMakingEnv` (paper III-E).

Uses the standard clipped-surrogate PPO objective (Schulman et al., 2017) --
the paper names PPO as its continuous-action-space training algorithm but
doesn't specify its own hyperparameters, so these are standard defaults, not
values taken from the paper.

Trains only the trunk + policy/value heads: the Attn-LOB backbone stays
frozen at its pretrained weights and its output is looked up from a
precomputed cache instead of recomputed per step (see `feature_cache.py`
and `networks.py`'s module docstrings) -- this is what keeps a training run
inside this replication's compute budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.distributions import Beta

from paper_replication.rl.env import MarketMakingEnv
from paper_replication.rl.networks import ContinuousActorCritic

Float32Array: TypeAlias = npt.NDArray[np.float32]


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    minibatch_size: int = 64
    episodes_per_update: int = 8
    n_updates: int = 50
    max_grad_norm: float = 0.5
    device: str = "cpu"
    seed: int = 0


@dataclass
class _Transition:
    lob_features: Float32Array
    dynamic_state: Float32Array
    agent_state: Float32Array
    action: Float32Array
    log_prob: float
    value: float
    reward: float
    done: bool


@torch.no_grad()
def _collect_episode(
    env: MarketMakingEnv,
    model: ContinuousActorCritic,
    lob_feature_cache: Float32Array,
    device: torch.device,
    episode_index: int,
) -> list[_Transition]:
    obs = env.reset(episode_index=episode_index)
    transitions: list[_Transition] = []
    done = False
    while not done:
        row = env.current_row
        lob_feat = lob_feature_cache[row]
        dyn = obs.dynamic_state.astype(np.float32)
        agent = obs.agent_state.astype(np.float32)

        lob_t = torch.from_numpy(lob_feat).unsqueeze(0).to(device)
        dyn_t = torch.from_numpy(dyn).unsqueeze(0).to(device)
        agent_t = torch.from_numpy(agent).unsqueeze(0).to(device)
        alpha, beta, value = model.forward_from_lob_features(lob_t, dyn_t, agent_t)
        dist = Beta(alpha, beta)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # type: ignore[no-untyped-call]

        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        result = env.step(action_np.astype(np.float64))

        transitions.append(
            _Transition(
                lob_features=lob_feat,
                dynamic_state=dyn,
                agent_state=agent,
                action=action_np,
                log_prob=float(log_prob.item()),
                value=float(value.item()),
                reward=result.reward,
                done=result.done,
            )
        )
        done = result.done
        if not done:
            assert result.observation is not None
            obs = result.observation
    return transitions


def _compute_gae(
    transitions: list[_Transition], gamma: float, gae_lambda: float
) -> tuple[Float32Array, Float32Array]:
    """GAE(lambda) advantages + returns. Never bootstraps across an episode boundary."""
    n = len(transitions)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = 0.0 if transitions[t].done else transitions[t + 1].value
        delta = transitions[t].reward + gamma * next_value - transitions[t].value
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[t] = last_gae
    values = np.array([t.value for t in transitions], dtype=np.float32)
    returns = advantages + values
    return advantages, returns


def train_ppo(
    env: MarketMakingEnv,
    model: ContinuousActorCritic,
    lob_feature_cache: Float32Array,
    config: PPOConfig,
    train_episode_indices: list[int] | None = None,
) -> list[dict[str, float]]:
    """Trains `model` in place; returns one metrics dict per update.

    `train_episode_indices`: episode indices to sample rollouts from
    (defaults to all of `env`'s episodes) -- pass the training portion of a
    `market_data.chronological_split` to keep held-out test episodes untouched.
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    model.to(device)

    indices = (
        list(range(env.n_episodes))
        if train_episode_indices is None
        else list(train_episode_indices)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, float]] = []
    for update in range(1, config.n_updates + 1):
        model.eval()
        all_transitions: list[_Transition] = []
        advantage_parts: list[Float32Array] = []
        return_parts: list[Float32Array] = []
        episode_rewards: list[float] = []

        for _ in range(config.episodes_per_update):
            episode_index = int(rng.choice(indices))
            episode = _collect_episode(
                env, model, lob_feature_cache, device, episode_index
            )
            episode_rewards.append(sum(t.reward for t in episode))
            advantages, returns = _compute_gae(episode, config.gamma, config.gae_lambda)
            all_transitions.extend(episode)
            advantage_parts.append(advantages)
            return_parts.append(returns)

        advantages_arr = np.concatenate(advantage_parts)
        returns_arr = np.concatenate(return_parts)
        advantages_arr = (advantages_arr - advantages_arr.mean()) / (
            advantages_arr.std() + 1e-8
        )

        lob_arr = np.stack([t.lob_features for t in all_transitions])
        dyn_arr = np.stack([t.dynamic_state for t in all_transitions])
        agent_arr = np.stack([t.agent_state for t in all_transitions])
        action_arr = np.stack([t.action for t in all_transitions])
        old_log_prob_arr = np.array(
            [t.log_prob for t in all_transitions], dtype=np.float32
        )

        n = len(all_transitions)
        model.train()
        total_policy_loss, total_value_loss, total_entropy, n_batches = 0.0, 0.0, 0.0, 0
        for _epoch in range(config.update_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, config.minibatch_size):
                idx = perm[start : start + config.minibatch_size]

                lob_t = torch.from_numpy(lob_arr[idx]).to(device)
                dyn_t = torch.from_numpy(dyn_arr[idx]).to(device)
                agent_t = torch.from_numpy(agent_arr[idx]).to(device)
                action_t = torch.from_numpy(action_arr[idx]).to(device)
                old_log_prob_t = torch.from_numpy(old_log_prob_arr[idx]).to(device)
                advantage_t = torch.from_numpy(advantages_arr[idx]).to(device)
                return_t = torch.from_numpy(returns_arr[idx]).to(device)

                alpha, beta, value = model.forward_from_lob_features(
                    lob_t, dyn_t, agent_t
                )
                dist = Beta(alpha, beta)
                new_log_prob = dist.log_prob(action_t).sum(  # type: ignore[no-untyped-call]
                    dim=-1
                )
                entropy = dist.entropy().sum(dim=-1).mean()  # type: ignore[no-untyped-call]

                ratio = torch.exp(new_log_prob - old_log_prob_t)
                surr1 = ratio * advantage_t
                surr2 = (
                    torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
                    * advantage_t
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value, return_t)
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                n_batches += 1

        history.append(
            {
                "update": float(update),
                "mean_episode_reward": float(np.mean(episode_rewards)),
                "policy_loss": total_policy_loss / n_batches,
                "value_loss": total_value_loss / n_batches,
                "entropy": total_entropy / n_batches,
            }
        )

    return history
