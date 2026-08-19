"""Evaluation metrics matching the paper's Table II (paper IV-C4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from paper_replication.rl.simulator import Fill


@dataclass
class EpisodeLog:
    """Per-episode record, holding enough data to compute Table II's metrics.

    final_pnl: the episode's final mark-to-market value minus its starting
        cash. The simulator's value starts at `RLConfig.initial_cash` each
        episode (paper IV-C2), so this *is* the episode's PnL.
    spreads: quoted `(ask_price - bid_price)` at each non-close step.
    abs_positions: `|inventory|` recorded at every step, including the
        forced-close step.
    fills: every fill executed during the episode (quoting + closing).
    """

    final_pnl: float
    spreads: list[float] = field(default_factory=list)
    abs_positions: list[float] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)

    @property
    def total_volume(self) -> float:
        return sum(fill.volume for fill in self.fills)

    @property
    def mean_abs_position(self) -> float:
        return float(np.mean(self.abs_positions)) if self.abs_positions else 0.0

    @property
    def mean_spread(self) -> float:
        return float(np.mean(self.spreads)) if self.spreads else 0.0


def nd_pnl(log: EpisodeLog) -> float:
    """ND-PnL [41]: PnL divided by the average quoted spread -- spreads captured."""
    if log.mean_spread == 0.0:
        return float("nan")
    return log.final_pnl / log.mean_spread


def pnlmap(log: EpisodeLog) -> float:
    """PnLMAP [44]: PnL per unit of mean absolute inventory."""
    if log.mean_abs_position == 0.0:
        return float("nan")
    return log.final_pnl / log.mean_abs_position


def profit_ratio(log: EpisodeLog) -> float:
    """PR: PnL divided by total trading volume."""
    if log.total_volume == 0.0:
        return float("nan")
    return log.final_pnl / log.total_volume


@dataclass
class AggregateMetrics:
    """Mean/std of each per-episode ratio across a set of evaluation episodes,
    plus a single Sharpe ratio computed from the episode-level PnL series.

    The paper reports "mean +/- std" per metric in Table II, most likely
    across repeated training seeds. This replication's compute budget
    doesn't support multi-seed training runs (see `docs/experiments.md`), so
    these means/stds are instead computed across a *single* trained policy's
    held-out test episodes: each episode contributes one ND-PnL/PnLMAP/PR
    sample. `sharpe` is computed once from the full episode-PnL series
    (mean / std of per-episode PnL, unannualized -- the paper doesn't
    specify an annualization convention).
    """

    n_episodes: int
    nd_pnl_mean: float
    nd_pnl_std: float
    pnlmap_mean: float
    pnlmap_std: float
    profit_ratio_mean: float
    profit_ratio_std: float
    sharpe: float
    mean_abs_position: float


def aggregate_metrics(logs: list[EpisodeLog]) -> AggregateMetrics:
    nd = np.array([nd_pnl(log) for log in logs])
    pm = np.array([pnlmap(log) for log in logs])
    pr = np.array([profit_ratio(log) for log in logs])
    pnl = np.array([log.final_pnl for log in logs])
    positions = np.array([log.mean_abs_position for log in logs])

    pnl_std = float(pnl.std(ddof=0))
    sharpe = float(pnl.mean() / pnl_std) if pnl_std > 0 else float("nan")

    return AggregateMetrics(
        n_episodes=len(logs),
        nd_pnl_mean=float(np.nanmean(nd)),
        nd_pnl_std=float(np.nanstd(nd)),
        pnlmap_mean=float(np.nanmean(pm)),
        pnlmap_std=float(np.nanstd(pm)),
        profit_ratio_mean=float(np.nanmean(pr)),
        profit_ratio_std=float(np.nanstd(pr)),
        sharpe=sharpe,
        mean_abs_position=float(positions.mean()) if len(positions) else 0.0,
    )
