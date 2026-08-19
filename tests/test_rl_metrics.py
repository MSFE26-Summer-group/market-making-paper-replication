"""Tests for paper_replication.rl.metrics."""

import math

from paper_replication.rl.metrics import (
    EpisodeLog,
    aggregate_metrics,
    nd_pnl,
    pnlmap,
    profit_ratio,
)
from paper_replication.rl.simulator import Fill


def test_nd_pnl_divides_by_mean_spread() -> None:
    log = EpisodeLog(final_pnl=10.0, spreads=[1.0, 2.0, 3.0])

    assert nd_pnl(log) == 5.0  # 10 / mean([1,2,3])=2


def test_nd_pnl_nan_when_no_spreads_recorded() -> None:
    log = EpisodeLog(final_pnl=10.0, spreads=[])

    assert math.isnan(nd_pnl(log))


def test_pnlmap_divides_by_mean_abs_position() -> None:
    log = EpisodeLog(final_pnl=8.0, abs_positions=[0.0, 4.0])

    assert pnlmap(log) == 4.0  # 8 / mean([0,4])=2


def test_pnlmap_nan_when_always_flat() -> None:
    log = EpisodeLog(final_pnl=8.0, abs_positions=[0.0, 0.0])

    assert math.isnan(pnlmap(log))


def test_profit_ratio_divides_by_total_volume() -> None:
    log = EpisodeLog(
        final_pnl=6.0,
        fills=[
            Fill(side=1, price=100.0, volume=1.0),
            Fill(side=-1, price=101.0, volume=2.0),
        ],
    )

    assert profit_ratio(log) == 2.0  # 6 / (1+2)


def test_profit_ratio_nan_when_no_fills() -> None:
    log = EpisodeLog(final_pnl=6.0, fills=[])

    assert math.isnan(profit_ratio(log))


def test_aggregate_metrics_over_multiple_episodes() -> None:
    logs = [
        EpisodeLog(final_pnl=10.0, spreads=[2.0], abs_positions=[1.0], fills=[]),
        EpisodeLog(final_pnl=-4.0, spreads=[2.0], abs_positions=[1.0], fills=[]),
    ]

    metrics = aggregate_metrics(logs)

    assert metrics.n_episodes == 2
    assert metrics.nd_pnl_mean == 1.5  # mean(10/2, -4/2) = mean(5, -2)
    assert metrics.mean_abs_position == 1.0


def test_aggregate_metrics_sharpe_is_nan_for_zero_variance_pnl() -> None:
    logs = [
        EpisodeLog(final_pnl=5.0, spreads=[1.0], abs_positions=[1.0]),
        EpisodeLog(final_pnl=5.0, spreads=[1.0], abs_positions=[1.0]),
    ]

    metrics = aggregate_metrics(logs)

    assert math.isnan(metrics.sharpe)


def test_aggregate_metrics_sharpe_is_positive_for_consistently_positive_pnl() -> None:
    logs = [
        EpisodeLog(final_pnl=5.0, spreads=[1.0], abs_positions=[1.0]),
        EpisodeLog(final_pnl=7.0, spreads=[1.0], abs_positions=[1.0]),
        EpisodeLog(final_pnl=6.0, spreads=[1.0], abs_positions=[1.0]),
    ]

    metrics = aggregate_metrics(logs)

    assert metrics.sharpe > 0
