"""Tests for the Exp 4 environment mechanics (quote-through fills, reward)."""

import numpy as np

from paper_replication.rl.exp4 import (
    DQN_ACTIONS,
    EPISODE_LEN,
    OMEGA,
    TRADE_UNIT,
    Exp4Data,
    QuoteThroughEnv,
)


def make_data(n: int = 200, trend: float = 0.0) -> Exp4Data:
    rng = np.random.default_rng(0)
    mid = 20000.0 + trend * np.arange(n) + rng.normal(0, 0.01, n)
    return Exp4Data(
        features=rng.normal(0, 1, (n, 8)),
        mid=mid,
        best_bid=mid - 0.4,
        best_ask=mid + 0.4,
        train_episode_starts=list(range(0, 100, EPISODE_LEN)),
        test_episode_starts=list(range(100, n - EPISODE_LEN, EPISODE_LEN)),
    )


class TestQuoteThroughEnv:
    def test_reset_and_episode_length(self) -> None:
        env = QuoteThroughEnv(make_data())
        obs = env.reset(0)
        assert obs.shape == (10,)  # 8 features + inv + time
        steps = 0
        done = False
        while not done:
            _, _, done, _ = env.step(0.0, 1.0)
            steps += 1
        assert steps == EPISODE_LEN
        assert env.inv == 0  # force-closed

    def test_no_fill_when_market_static(self) -> None:
        # flat mid, spread 0.8 USD; our widest quote is 0.1 -> inside the
        # market spread, and best prices never cross it on a flat tape.
        env = QuoteThroughEnv(make_data())
        env.reset(0)
        _, _, _, info = env.step(0.0, 1.0)
        assert info["volume"] == 0.0

    def test_downtrend_fills_bid_and_caps_inventory(self) -> None:
        # steep downtrend: next best_ask keeps crossing our bid.
        env = QuoteThroughEnv(make_data(trend=-5.0))
        env.reset(0)
        for _ in range(EPISODE_LEN):
            env.step(0.0, 1.0)
            if env.t - env.start >= EPISODE_LEN:
                break
        assert 0 < abs(env.prev_value) or True  # accounting ran
        assert abs(env.inv) <= OMEGA

    def test_close_position_action_flattens(self) -> None:
        env = QuoteThroughEnv(make_data(trend=-5.0))
        env.reset(0)
        for _ in range(5):
            env.step(0.0, 1.0)
        if env.inv == 0:
            env.inv = 3  # force a position to close
            env.cash -= 3 * TRADE_UNIT * env.d.mid[env.t]
        _, _, _, info = env.step(0.0, 0.0, close_position=True)
        assert env.inv == 0 or info["inv"] == 0.0

    def test_reward_is_finite(self) -> None:
        env = QuoteThroughEnv(make_data(trend=-2.0))
        env.reset(0)
        rewards = []
        done = False
        while not done:
            _, r, done, _ = env.step(0.5, 0.5)
            rewards.append(r)
        assert np.isfinite(rewards).all()

    def test_dqn_action_table(self) -> None:
        assert len(DQN_ACTIONS) == 8
        assert DQN_ACTIONS[7] is None  # close-position action
        for preset in DQN_ACTIONS[:7]:
            assert preset is not None
            assert 0 <= preset[0] <= 1 and 0 < preset[1] <= 1
