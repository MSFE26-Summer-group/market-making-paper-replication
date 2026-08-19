"""Runs a policy over evaluation episodes and aggregates Table II metrics (paper IV-C4).

`Policy` is deliberately minimal -- `(observation, row) -> action` -- so
trained networks (via a thin wrapper), the baselines in `baselines.py`, and
ad-hoc lambdas in tests/notebooks can all be evaluated through the exact
same harness and metrics.
"""

from __future__ import annotations

from typing import Callable, TypeAlias, cast

from paper_replication.rl.env import Action, MarketMakingEnv, Observation
from paper_replication.rl.metrics import AggregateMetrics, EpisodeLog, aggregate_metrics
from paper_replication.rl.simulator import Fill

Policy: TypeAlias = Callable[[Observation, int], Action]


def run_episode(env: MarketMakingEnv, episode_index: int, policy: Policy) -> EpisodeLog:
    """Runs one episode under `policy`.

    Returns enough data to compute Table II's metrics.

    `policy(observation, row)` is called once per non-terminal step;
    `env`'s own `action_space_kind` determines what it must return (a
    continuous `(2,)` array or a discrete `int`).
    """
    obs = env.reset(episode_index=episode_index)
    spreads: list[float] = []
    abs_positions = [abs(env.simulator.inventory)]
    fills: list[Fill] = []
    result = None
    done = False

    while not done:
        row = env.current_row
        action = policy(obs, row)
        result = env.step(action)

        quote = result.info.get("quote")
        if (
            quote is not None
            and not quote.close_position  # type: ignore[attr-defined]
            and quote.bid_price is not None  # type: ignore[attr-defined]
            and quote.ask_price is not None  # type: ignore[attr-defined]
        ):
            spreads.append(quote.ask_price - quote.bid_price)  # type: ignore[attr-defined]
        fills.extend(cast(list[Fill], result.info["fills"]))
        abs_positions.append(abs(cast(float, result.info["inventory"])))

        done = result.done
        if not done:
            assert result.observation is not None
            obs = result.observation

    assert result is not None
    final_pnl = cast(float, result.info["value"]) - env.config.initial_cash
    return EpisodeLog(
        final_pnl=final_pnl, spreads=spreads, abs_positions=abs_positions, fills=fills
    )


def evaluate_policy(
    env: MarketMakingEnv, policy: Policy, episode_indices: list[int]
) -> AggregateMetrics:
    """Runs `policy` over `episode_indices` and aggregates Table II's metrics."""
    logs = [run_episode(env, i, policy) for i in episode_indices]
    return aggregate_metrics(logs)
