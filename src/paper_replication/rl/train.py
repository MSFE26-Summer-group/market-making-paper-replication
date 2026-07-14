"""Minimal PPO training loop for the market-making agent.

Run as a script for a quick sample training on the BTC-USDT data:

    uv run python -m paper_replication.rl.train --updates 20
"""

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from paper_replication.rl.data import build_dataset, load_lob
from paper_replication.rl.env import EnvConfig, MarketMakingEnv
from paper_replication.rl.model import ActorCritic


@dataclass
class PPOConfig:
    rollout_episodes: int = 4
    epochs: int = 4
    minibatch: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, cfg: PPOConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation over concatenated episodes."""
    adv = np.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + cfg.gamma * next_value * nonterminal - values[t]
        last = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last
        adv[t] = last
    returns = adv + values[: len(adv)]
    return adv, returns


def collect_rollout(
    env: MarketMakingEnv, model: ActorCritic, cfg: PPOConfig
) -> dict[str, Tensor]:
    """Run episodes with the current policy and stack the transitions."""
    buf: dict[str, list[np.ndarray | float]] = {
        k: []
        for k in ("states", "aux", "actions", "logp", "values", "rewards", "dones")
    }
    ep_rewards: list[float] = []
    for _ in range(cfg.rollout_episodes):
        res = env.reset()
        ep_total = 0.0
        while not res.done:
            s = torch.from_numpy(res.state).unsqueeze(0)
            a = torch.from_numpy(res.aux).unsqueeze(0)
            action, logp, value = model.act(s, a)
            buf["states"].append(res.state)
            buf["aux"].append(res.aux)
            buf["actions"].append(action.numpy()[0])
            buf["logp"].append(float(logp))
            buf["values"].append(float(value))
            res = env.step(action.numpy()[0])
            buf["rewards"].append(res.reward)
            buf["dones"].append(float(res.done))
            ep_total += res.reward
        ep_rewards.append(ep_total)

    adv, returns = compute_gae(
        np.array(buf["rewards"], dtype=np.float32),
        np.array(buf["values"], dtype=np.float32),
        np.array(buf["dones"], dtype=np.float32),
        cfg,
    )
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    out = {
        "states": torch.from_numpy(np.array(buf["states"], dtype=np.float32)),
        "aux": torch.from_numpy(np.array(buf["aux"], dtype=np.float32)),
        "actions": torch.from_numpy(np.array(buf["actions"], dtype=np.float32)),
        "logp": torch.tensor(buf["logp"], dtype=torch.float32),
        "adv": torch.from_numpy(adv),
        "returns": torch.from_numpy(returns),
    }
    out["ep_reward_mean"] = torch.tensor(float(np.mean(ep_rewards)))
    return out


def ppo_update(
    model: ActorCritic,
    optim: torch.optim.Optimizer,
    batch: dict[str, Tensor],
    cfg: PPOConfig,
) -> float:
    """Clipped-objective PPO epochs over the rollout; returns last loss."""
    n = len(batch["logp"])
    loss = torch.tensor(0.0)
    for _ in range(cfg.epochs):
        for idx in torch.split(torch.randperm(n), cfg.minibatch):
            dist, value = model(batch["states"][idx], batch["aux"][idx])
            logp = dist.log_prob(batch["actions"][idx]).sum(-1)
            ratio = (logp - batch["logp"][idx]).exp()
            adv = batch["adv"][idx]
            pg = -torch.min(
                ratio * adv,
                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv,
            ).mean()
            vf = (value - batch["returns"][idx]).pow(2).mean()
            ent = dist.entropy().sum(-1).mean()
            loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
            optim.zero_grad()
            loss.backward()
            optim.step()
    return float(loss.detach())


def train(
    data_path: str, updates: int = 20, seed: int = 0
) -> tuple[ActorCritic, list[float]]:
    """Train a sample agent; returns the model and per-update rewards."""
    torch.manual_seed(seed)
    ds = build_dataset(load_lob(data_path), window=50)
    env = MarketMakingEnv(ds, EnvConfig(), rng=np.random.default_rng(seed))
    model = ActorCritic()
    optim = torch.optim.Adam(model.parameters(), lr=PPOConfig.lr)
    cfg = PPOConfig()

    history: list[float] = []
    for u in range(updates):
        batch = collect_rollout(env, model, cfg)
        loss = ppo_update(model, optim, batch, cfg)
        mean_r = float(batch["ep_reward_mean"])
        history.append(mean_r)
        print(
            f"update {u + 1:>3}/{updates} | ep_reward {mean_r:>10.2f} | loss {loss:.4f}"
        )
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/btc_usdt_20221019_20221030_lob.parquet")
    parser.add_argument("--updates", type=int, default=20)
    args = parser.parse_args()
    train(args.data, updates=args.updates)
