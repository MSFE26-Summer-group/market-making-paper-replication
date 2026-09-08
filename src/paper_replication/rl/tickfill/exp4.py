"""Experiment 4: RL market making (C-PPO vs D-DQN vs baselines).

Implements the report's Section 9 specification:

- Chronological 80/20 split over the 88,678 windowed rows; episodes are
  consecutive blocks of 30 non-overlapping steps (2,364 train / 591 test).
- Frozen pretrained Attn-LOB backbone; its 192-d features are precomputed
  once for every window (the "feature cache") and the RL loop indexes into
  that cache. Agent state = backbone features + (inventory/omega, step/30).
- Quote-through fills: a bid executes iff the NEXT snapshot's best ask has
  crossed it (best_ask <= bid); an ask iff next best_bid >= ask. One
  minimum-trade-unit (0.001 BTC) per fill, position capped at +/- omega
  units.
- Paper action space: A1, A2 in [0,1]; delta = A1 * max_bias;
  p_r = mid - sign(inv) * delta; spread = A2 * max_spread;
  quotes = p_r +/- spread/2 (paper eq. 8-11, max_bias=0.05, max_spread=0.1
  in USD as the report states them).
- Paper reward (eq. 12-16): R = DP + TP - IP with eta=0.5, zeta=0.01,
  IP on inventory in trade units; episode-end force close at mid.
- C-PPO: Beta(alpha, beta) per action dim, 300 updates x 16 episodes,
  4 epochs, minibatch 128. D-DQN: dueling heads over 8 discrete actions,
  150k steps, replay 50k, eps 1.0 -> 0.05 over 100k steps, target sync
  every 1k steps. Single seed, no hyperparameter search (report 9.2).

Choices the report leaves unspecified are documented inline; each is a
plain literature default, not tuned.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.distributions import Beta

from paper_replication.features.lob_state import (
    load_lob_snapshot,
    lob_state_matrix,
    normalize_lob_state,
    rolling_windows,
)
from paper_replication.models.attn_lob import AttnLOB, AttnLOBConfig

FloatArray = npt.NDArray[np.float64]

EPISODE_LEN = 30
OMEGA = 10  # max inventory in trade units
ETA = 0.5
ZETA = 0.01
MAX_BIAS = 0.05  # USD, as stated in the report
MAX_SPREAD = 0.1  # USD
# Paper mirror: its Fixed baselines quote at LOB levels 1-3 and the A-share
# tick grid (~1 tick = the whole market spread) leaves no room inside the
# touch, so quoting inside the spread effectively does not exist there.
# With TOUCH_FLOOR on, quotes are clamped to the current touch (level 1).
TOUCH_FLOOR = False
TRADE_UNIT = 0.001  # BTC


# --------------------------------------------------------------------------
# Market data + feature cache
# --------------------------------------------------------------------------
@dataclass
class Exp4Data:
    features: FloatArray  # (n, 192) frozen backbone features
    mid: FloatArray
    best_bid: FloatArray
    best_ask: FloatArray
    train_episode_starts: list[int]
    test_episode_starts: list[int]
    sell_min: FloatArray | None = None  # per-interval seller-initiated min (tick fills)
    buy_max: FloatArray | None = None  # per-interval buyer-initiated max


def build_exp4_data(
    lob_parquet: str,
    checkpoint_path: str,
    device: str = "cpu",
    n_levels: int = 10,
    window_T: int = 50,
    ticks_parquet: str | None = None,
    ts_min: float | None = None,
    ts_max: float | None = None,
    split_ts: float | None = None,
) -> Exp4Data:
    """Load snapshots, run the frozen backbone once, and form episode grids.

    ts_min/ts_max crop the snapshot file to [ts_min, ts_max) before windowing;
    split_ts puts the chronological train/test boundary at a fixed timestamp
    (rows with aligned timestamp >= split_ts are test) instead of the default
    80/20 row split — used for fixed-calendar-test comparisons.
    """
    df = load_lob_snapshot(lob_parquet, symbol="BTCUSDT", n_levels=n_levels)
    if ts_min is not None:
        df = df[df["timestamp"] >= ts_min]
    if ts_max is not None:
        df = df[df["timestamp"] < ts_max]
    df = df.reset_index(drop=True)
    raw = lob_state_matrix(df, "BTCUSDT", n_levels)
    windows = rolling_windows(raw, window_T)
    norm = normalize_lob_state(windows, n_levels)  # (n_win, 50, 40)

    model = AttnLOB(AttnLOBConfig(dropout=0.3))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state)
    model.to(device).eval()

    feats = np.empty((len(norm), model.embed_dim), dtype=np.float64)
    with torch.no_grad():
        for lo in range(0, len(norm), 512):
            batch = torch.from_numpy(norm[lo : lo + 512]).float().to(device)
            feats[lo : lo + 512] = model.forward_features(batch).cpu().numpy()

    off = window_T - 1  # align row i of feats with df row i + off
    mid = df["mid_price"].to_numpy(dtype=np.float64)[off:]
    bb = df[f"BTCUSDT.bid_price_1"].to_numpy(dtype=np.float64)[off:]
    ba = df[f"BTCUSDT.ask_price_1"].to_numpy(dtype=np.float64)[off:]

    n = len(feats)
    if split_ts is None:
        n_train_rows = int(n * 0.8)
    else:
        ts_aligned = df["timestamp"].to_numpy(dtype=np.float64)[off:]
        n_train_rows = int(np.searchsorted(ts_aligned, split_ts, side="left"))
    train_starts = list(range(0, n_train_rows - EPISODE_LEN, EPISODE_LEN))
    test_starts = list(range(n_train_rows, n - EPISODE_LEN, EPISODE_LEN))

    sell_min = buy_max = None
    if ticks_parquet is not None:
        import pandas as pd

        ticks = pd.read_parquet(
            ticks_parquet, columns=["timestamp", "price", "side"]
        ).sort_values("timestamp")
        ts_all = df["timestamp"].to_numpy()
        idx = np.searchsorted(ts_all, ticks["timestamp"].to_numpy(), side="left")
        valid = (idx > 0) & (idx < len(ts_all))
        t = ticks.iloc[valid].assign(row=idx[valid])
        smin = t[t["side"] == -1].groupby("row")["price"].min()
        bmax = t[t["side"] == 1].groupby("row")["price"].max()
        sell_min = np.full(len(ts_all), np.nan)
        buy_max = np.full(len(ts_all), np.nan)
        sell_min[smin.index] = smin.to_numpy()
        buy_max[bmax.index] = bmax.to_numpy()
        sell_min, buy_max = sell_min[off:], buy_max[off:]
    return Exp4Data(feats, mid, bb, ba, train_starts, test_starts, sell_min, buy_max)


# --------------------------------------------------------------------------
# Environment (quote-through fills, paper reward)
# --------------------------------------------------------------------------
class QuoteThroughEnv:
    """30-step episodic market-making env on the cached feature grid."""

    def __init__(self, data: Exp4Data, fill_mode: str = "quote_through") -> None:
        if fill_mode == "tick" and data.sell_min is None:
            raise ValueError("tick fill_mode requires ticks_parquet in build_exp4_data")
        self.fill_mode = fill_mode
        self.d = data
        self.t = 0
        self.start = 0
        self.inv = 0  # in trade units
        self.cash = 0.0
        self.prev_value = 0.0

    def obs(self) -> FloatArray:
        aux = np.array(
            [self.inv / OMEGA, (self.t - self.start) / EPISODE_LEN], dtype=np.float64
        )
        return np.concatenate([self.d.features[self.t], aux])

    def reset(self, start: int) -> FloatArray:
        self.start = start
        self.t = start
        self.inv = 0
        self.cash = 0.0
        self.prev_value = 0.0
        return self.obs()

    def step(
        self, a1: float, a2: float, close_position: bool = False
    ) -> tuple[FloatArray, float, bool, dict[str, float]]:
        mid = self.d.mid[self.t]
        nxt = self.t + 1
        tp = 0.0
        volume = 0.0

        if close_position:
            # paper's discrete action 7: flatten with a market order at the
            # touch (buy at ask / sell at bid).
            if self.inv > 0:
                px = self.d.best_bid[self.t]
                self.cash += self.inv * TRADE_UNIT * px
                tp += self.inv * TRADE_UNIT * (px - mid)
                volume += abs(self.inv) * TRADE_UNIT * px
                self.inv = 0
            elif self.inv < 0:
                px = self.d.best_ask[self.t]
                self.cash += self.inv * TRADE_UNIT * px
                tp += -self.inv * TRADE_UNIT * (mid - px)
                volume += abs(self.inv) * TRADE_UNIT * px
                self.inv = 0
        else:
            delta = float(np.clip(a1, 0, 1)) * MAX_BIAS
            spread = float(np.clip(a2, 0, 1)) * MAX_SPREAD
            p_r = mid - float(np.sign(self.inv)) * delta
            bid = p_r - spread / 2
            ask = p_r + spread / 2
            if TOUCH_FLOOR:
                bid = min(bid, self.d.best_bid[self.t])
                ask = max(ask, self.d.best_ask[self.t])

            if self.fill_mode == "tick":
                assert self.d.sell_min is not None and self.d.buy_max is not None
                sm, bm = self.d.sell_min[nxt], self.d.buy_max[nxt]
                bid_hit = bool(sm == sm and sm <= bid)  # NaN-safe
                ask_hit = bool(bm == bm and bm >= ask)
            else:
                bid_hit = self.d.best_ask[nxt] <= bid
                ask_hit = self.d.best_bid[nxt] >= ask
            if bid_hit and self.inv < OMEGA:
                self.inv += 1
                self.cash -= TRADE_UNIT * bid
                tp += TRADE_UNIT * (self.d.mid[nxt] - bid)
                volume += TRADE_UNIT * bid
            if ask_hit and self.inv > -OMEGA:
                self.inv -= 1
                self.cash += TRADE_UNIT * ask
                tp += TRADE_UNIT * (ask - self.d.mid[nxt])
                volume += TRADE_UNIT * ask

        value = self.cash + self.inv * TRADE_UNIT * self.d.mid[nxt]
        dpnl = value - self.prev_value
        dp = dpnl - max(0.0, ETA * dpnl)  # paper eq. 13
        ip = ZETA * float(self.inv) ** 2  # paper eq. 15
        reward = dp + tp - ip  # paper eq. 16
        self.prev_value = value

        self.t = nxt
        done = self.t - self.start >= EPISODE_LEN
        if done:  # force close at mid
            self.cash += self.inv * TRADE_UNIT * self.d.mid[self.t]
            self.inv = 0
        return self.obs(), reward, done, {"volume": volume, "inv": float(self.inv)}


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------
class BetaActorCritic(nn.Module):
    """Trunk + Beta(alpha, beta) policy per action dim + value head.

    Trunk width 64 and lr 3e-4 are literature defaults (unspecified in the
    report). softplus(x) + 1 keeps both Beta parameters > 1 (unimodal).
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh())
        self.ab = nn.Linear(64, 4)  # (alpha1, beta1, alpha2, beta2)
        self.v = nn.Linear(64, 1)

    def dist_value(self, x: Tensor) -> tuple[Beta, Tensor]:
        h = self.trunk(x)
        ab = nn.functional.softplus(self.ab(h)) + 1.0
        dist = Beta(ab[..., 0::2], ab[..., 1::2])
        return dist, self.v(h).squeeze(-1)


class DuelingQNet(nn.Module):
    """Dueling value/advantage heads over 8 discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int = 8) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU())
        self.val = nn.Linear(64, 1)
        self.adv = nn.Linear(64, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        h = self.trunk(x)
        v, a = self.val(h), self.adv(h)
        q: Tensor = v + a - a.mean(dim=-1, keepdim=True)
        return q


# The paper's 8 discrete actions are (spread, bias) presets plus a
# close-position action; the exact presets are not published, so we use an
# even grid over the same caps (documented deviation).
DQN_ACTIONS: list[tuple[float, float] | None] = [
    (0.0, 1.0 / 3),
    (0.0, 2.0 / 3),
    (0.0, 1.0),
    (0.5, 2.0 / 3),
    (0.5, 1.0),
    (1.0, 2.0 / 3),
    (1.0, 1.0),
    None,  # action 7: close position with a market order
]


def train_cppo(
    data: Exp4Data,
    seed: int = 0,
    updates: int = 300,
    episodes_per_update: int = 16,
    fill_mode: str = "quote_through",
) -> BetaActorCritic:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = QuoteThroughEnv(data, fill_mode)
    obs_dim = data.features.shape[1] + 2
    model = BetaActorCritic(obs_dim)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    for _u in range(updates):
        obs_b, act_b, logp_b, rew_b, val_b, done_b = [], [], [], [], [], []
        for _e in range(episodes_per_update):
            start = int(rng.choice(data.train_episode_starts))
            obs = env.reset(start)
            done = False
            while not done:
                x = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad():
                    dist, value = model.dist_value(x)
                    a = dist.sample()
                    logp = dist.log_prob(a).sum(-1)  # type: ignore[no-untyped-call]
                obs_b.append(obs)
                act_b.append(a.numpy()[0])
                logp_b.append(float(logp))
                val_b.append(float(value))
                a_np = a.numpy()[0]
                obs, r, done, _ = env.step(float(a_np[0]), float(a_np[1]))
                rew_b.append(r)
                done_b.append(float(done))

        rewards = np.array(rew_b, dtype=np.float64)
        values = np.array(val_b, dtype=np.float64)
        dones = np.array(done_b, dtype=np.float64)
        adv = np.zeros_like(rewards)
        last = 0.0
        for t in reversed(range(len(rewards))):
            nonterm = 1.0 - dones[t]
            nv = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + 0.99 * nv * nonterm - values[t]
            last = delta + 0.99 * 0.95 * nonterm * last
            adv[t] = last
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.from_numpy(np.array(obs_b)).float()
        act_t = torch.from_numpy(np.array(act_b)).float().clamp(1e-4, 1 - 1e-4)
        logp_t = torch.tensor(logp_b).float()
        adv_t = torch.from_numpy(adv).float()
        ret_t = torch.from_numpy(returns).float()

        n = len(obs_t)
        for _ep in range(4):
            for idx in torch.split(torch.randperm(n), 128):
                dist, value = model.dist_value(obs_t[idx])
                logp = dist.log_prob(act_t[idx]).sum(-1)  # type: ignore[no-untyped-call]
                ratio = (logp - logp_t[idx]).exp()
                a_ = adv_t[idx]
                pg = -torch.min(ratio * a_, ratio.clamp(0.8, 1.2) * a_).mean()
                vf = (value - ret_t[idx]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()  # type: ignore[no-untyped-call]
                loss = pg + 0.5 * vf - 0.01 * ent
                optim.zero_grad()
                loss.backward()
                optim.step()
    return model


def train_ddqn(
    data: Exp4Data,
    seed: int = 0,
    total_steps: int = 150_000,
    fill_mode: str = "quote_through",
) -> DuelingQNet:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = QuoteThroughEnv(data, fill_mode)
    obs_dim = data.features.shape[1] + 2
    q = DuelingQNet(obs_dim)
    q_target = DuelingQNet(obs_dim)
    q_target.load_state_dict(q.state_dict())
    optim = torch.optim.Adam(q.parameters(), lr=3e-4)
    replay: deque[tuple[FloatArray, int, float, FloatArray, float]] = deque(
        maxlen=50_000
    )

    obs = env.reset(int(rng.choice(data.train_episode_starts)))
    for step in range(total_steps):
        eps = max(0.05, 1.0 - 0.95 * step / 100_000)
        if rng.random() < eps:
            action = int(rng.integers(0, 8))
        else:
            with torch.no_grad():
                action = int(q(torch.from_numpy(obs).float()).argmax())
        preset = DQN_ACTIONS[action]
        if preset is None:
            nobs, r, done, _ = env.step(0.0, 0.0, close_position=True)
        else:
            nobs, r, done, _ = env.step(preset[0], preset[1])
        replay.append((obs, action, r, nobs, float(done)))
        obs = env.reset(int(rng.choice(data.train_episode_starts))) if done else nobs

        if len(replay) >= 1_000:
            idx = rng.integers(0, len(replay), 128)
            batch = [replay[int(i)] for i in idx]
            s = torch.from_numpy(np.array([b[0] for b in batch])).float()
            a = torch.tensor([b[1] for b in batch])
            r_ = torch.tensor([b[2] for b in batch]).float()
            s2 = torch.from_numpy(np.array([b[3] for b in batch])).float()
            d = torch.tensor([b[4] for b in batch]).float()
            with torch.no_grad():
                # double-DQN target: online net picks, target net evaluates
                a2 = q(s2).argmax(dim=1)
                tq = r_ + 0.99 * (1 - d) * q_target(s2).gather(
                    1, a2.unsqueeze(1)
                ).squeeze(1)
            pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, tq)
            optim.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optim.step()
        if step % 1_000 == 0:
            q_target.load_state_dict(q.state_dict())
    return q


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
Policy = Callable[[FloatArray, QuoteThroughEnv], tuple[float, float, bool]]


def evaluate_policy(
    data: Exp4Data, policy: Policy, fill_mode: str = "quote_through"
) -> dict[str, float]:
    env = QuoteThroughEnv(data, fill_mode)
    pnls, nd, pmap, pr = [], [], [], []
    for start in data.test_episode_starts:
        obs = env.reset(start)
        done = False
        inv_path: list[float] = []
        vol = 0.0
        while not done:
            a1, a2, close = policy(obs, env)
            obs, _r, done, info = env.step(a1, a2, close_position=close)
            inv_path.append(abs(info["inv"]) * TRADE_UNIT)
            vol += info["volume"]
        pnl = env.cash
        pnls.append(pnl)
        spread_mean = float(
            np.mean(
                data.best_ask[start : start + EPISODE_LEN]
                - data.best_bid[start : start + EPISODE_LEN]
            )
        )
        nd.append(pnl / spread_mean)
        map_ = float(np.mean(inv_path))
        pmap.append(pnl / map_ if map_ > 0 else 0.0)
        pr.append(pnl / vol if vol > 0 else 0.0)

    p = np.array(pnls)
    return {
        "nd_pnl_mean": float(np.mean(nd)),
        "nd_pnl_std": float(np.std(nd)),
        "pnlmap_mean": float(np.mean(pmap)),
        "pnlmap_std": float(np.std(pmap)),
        "profit_ratio_mean": float(np.mean(pr)),
        "sharpe": float(p.mean() / p.std()) if p.std() > 0 else 0.0,
        "pnl_mean_usd": float(p.mean()),
    }


def make_policies(
    data: Exp4Data, cppo: BetaActorCritic, ddqn: DuelingQNet, seed: int = 1
) -> dict[str, Policy]:
    rng = np.random.default_rng(seed)

    def cppo_policy(obs: FloatArray, env: QuoteThroughEnv) -> tuple[float, float, bool]:
        with torch.no_grad():
            dist, _ = cppo.dist_value(torch.from_numpy(obs).float().unsqueeze(0))
            m = dist.mean.numpy()[0]  # deterministic eval
        return float(m[0]), float(m[1]), False

    def ddqn_policy(obs: FloatArray, env: QuoteThroughEnv) -> tuple[float, float, bool]:
        with torch.no_grad():
            a = int(ddqn(torch.from_numpy(obs).float()).argmax())
        preset = DQN_ACTIONS[a]
        if preset is None:
            return 0.0, 0.0, True
        return preset[0], preset[1], False

    def random_policy(
        obs: FloatArray, env: QuoteThroughEnv
    ) -> tuple[float, float, bool]:
        return float(rng.random()), float(rng.random()), False

    def fixed(frac: float) -> Policy:
        def p(obs: FloatArray, env: QuoteThroughEnv) -> tuple[float, float, bool]:
            return 0.0, frac, False

        return p

    # AS-inspired baseline (report 9.2): realized volatility over the last
    # 30 steps stands in for the calibrated sigma; gamma anchored so one
    # unit of inventory skews by MAX_BIAS/OMEGA (documented adaptation).
    def as_policy(obs: FloatArray, env: QuoteThroughEnv) -> tuple[float, float, bool]:
        t = env.t
        lo = max(1, t - 30)
        rets = np.diff(data.mid[lo : t + 1])
        sigma2 = float(np.var(rets)) if len(rets) > 1 else 0.0
        t_rem = EPISODE_LEN - (t - env.start)
        gamma = (MAX_BIAS / OMEGA) / (sigma2 * EPISODE_LEN + 1e-12)
        bias = min(1.0, abs(env.inv) * gamma * sigma2 * t_rem / MAX_BIAS)
        spread = min(
            1.0,
            (gamma * sigma2 * t_rem + 2 / max(gamma, 1e-12) * np.log1p(gamma / 1.5))
            / MAX_SPREAD,
        )
        return bias, spread, False

    return {
        "C-PPO": cppo_policy,
        "D-DQN": ddqn_policy,
        "Random": random_policy,
        "Fixed(15%)": fixed(0.15),
        "Fixed(50%)": fixed(0.5),
        "Fixed(100%)": fixed(1.0),
        "AS": as_policy,
    }


def run_exp4(
    lob_parquet: str,
    checkpoint: str,
    out_dir: str,
    device: str = "cpu",
    fill_mode: str = "quote_through",
    ticks_parquet: str | None = None,
    ts_min: float | None = None,
    ts_max: float | None = None,
    split_ts: float | None = None,
    cppo_updates: int = 300,
    ddqn_steps: int = 150_000,
    max_bias: float | None = None,
    max_spread: float | None = None,
    touch_floor: bool | None = None,
) -> dict[str, dict[str, float]]:
    # Caps are module-level constants read at step() time; overriding them
    # here (scale-transfer reruns) affects every env/policy built below.
    global MAX_BIAS, MAX_SPREAD, TOUCH_FLOOR
    if max_bias is not None:
        MAX_BIAS = max_bias
    if max_spread is not None:
        MAX_SPREAD = max_spread
    if touch_floor is not None:
        TOUCH_FLOOR = touch_floor
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_exp4_data(
        lob_parquet,
        checkpoint,
        device=device,
        ticks_parquet=ticks_parquet,
        ts_min=ts_min,
        ts_max=ts_max,
        split_ts=split_ts,
    )
    print(
        f"data ready: {len(data.features)} rows, "
        f"{len(data.train_episode_starts)} train / {len(data.test_episode_starts)} test episodes",
        flush=True,
    )
    cppo = train_cppo(data, fill_mode=fill_mode, updates=cppo_updates)
    print("C-PPO trained", flush=True)
    ddqn = train_ddqn(data, fill_mode=fill_mode, total_steps=ddqn_steps)
    print("D-DQN trained", flush=True)
    results = {
        name: evaluate_policy(data, pol, fill_mode)
        for name, pol in make_policies(data, cppo, ddqn).items()
    }
    (out / "exp4_metrics.json").write_text(json.dumps(results, indent=2))
    torch.save(cppo.state_dict(), out / "exp4_cppo.pt")
    torch.save(ddqn.state_dict(), out / "exp4_ddqn.pt")
    return results
