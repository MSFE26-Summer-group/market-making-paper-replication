"""Tests for paper_replication.rl.reward."""

import pytest

from paper_replication.rl.reward import (
    dampened_pnl,
    delta_pnl,
    hybrid_reward,
    inventory_punishment,
    trading_pnl,
)
from paper_replication.rl.simulator import Fill


def test_delta_pnl_is_the_change_in_value() -> None:
    assert delta_pnl(value=110.0, previous_value=100.0) == 10.0


def test_dampened_pnl_leaves_losses_untouched() -> None:
    assert dampened_pnl(pnl_delta=-10.0, eta=0.5) == -10.0


def test_dampened_pnl_shrinks_gains() -> None:
    # DP = delta - max(0, eta*delta) = 10 - 0.5*10 = 5
    assert dampened_pnl(pnl_delta=10.0, eta=0.5) == pytest.approx(5.0)


def test_trading_pnl_rewards_buying_below_mid_price() -> None:
    fills = [Fill(side=1, price=99.0, volume=2.0)]

    assert trading_pnl(fills, mid_price=100.0) == pytest.approx(2.0)


def test_trading_pnl_rewards_selling_above_mid_price() -> None:
    fills = [Fill(side=-1, price=101.0, volume=2.0)]

    assert trading_pnl(fills, mid_price=100.0) == pytest.approx(2.0)


def test_trading_pnl_penalizes_unfavorable_fills() -> None:
    fills = [Fill(side=1, price=101.0, volume=1.0)]

    assert trading_pnl(fills, mid_price=100.0) == pytest.approx(-1.0)


def test_trading_pnl_sums_multiple_fills() -> None:
    fills = [
        Fill(side=1, price=99.0, volume=1.0),
        Fill(side=-1, price=101.0, volume=1.0),
    ]

    assert trading_pnl(fills, mid_price=100.0) == pytest.approx(2.0)


def test_trading_pnl_empty_fills_is_zero() -> None:
    assert trading_pnl([], mid_price=100.0) == 0.0


def test_inventory_punishment_is_quadratic() -> None:
    assert inventory_punishment(inventory=3.0, zeta=0.01) == pytest.approx(0.09)
    assert inventory_punishment(inventory=-3.0, zeta=0.01) == pytest.approx(0.09)


def test_hybrid_reward_combines_all_three_terms() -> None:
    fills = [Fill(side=1, price=99.0, volume=1.0)]

    reward = hybrid_reward(
        pnl_delta=10.0,
        fills=fills,
        mid_price=100.0,
        inventory=2.0,
        eta=0.5,
        zeta=0.01,
    )

    # DP=5, TP=1, IP=0.04 -> 5 + 1 - 0.04 = 5.96
    assert reward == pytest.approx(5.96)
