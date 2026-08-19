"""Tests for paper_replication.rl.simulator."""

from paper_replication.rl.simulator import MarketSimulator


def _sim(min_unit: float = 1.0, max_inv: float = 3.0) -> MarketSimulator:
    return MarketSimulator(minimum_trade_unit=min_unit, max_inventory=max_inv)


def test_reset_zeroes_inventory_and_sets_cash() -> None:
    sim = _sim()
    sim.cash = 999.0
    sim.inventory = 2.0

    sim.reset(cash=100.0)

    assert sim.cash == 100.0
    assert sim.inventory == 0.0


def test_value_is_cash_plus_inventory_times_mid_price() -> None:
    sim = _sim()
    sim.reset(cash=50.0)
    sim.inventory = 2.0

    assert sim.value(mid_price=10.0) == 70.0


def test_bid_fills_when_it_crosses_next_best_ask() -> None:
    sim = _sim()
    sim.reset(cash=0.0)

    fills = sim.step(
        bid_price=101.0, ask_price=None, next_best_bid=99.0, next_best_ask=100.0
    )

    assert len(fills) == 1
    assert fills[0].side == 1
    assert fills[0].price == 101.0
    assert fills[0].volume == 1.0
    assert sim.inventory == 1.0
    assert sim.cash == -101.0


def test_ask_fills_when_it_crosses_next_best_bid() -> None:
    sim = _sim()
    sim.reset(cash=0.0)

    fills = sim.step(
        bid_price=None, ask_price=99.0, next_best_bid=100.0, next_best_ask=101.0
    )

    assert len(fills) == 1
    assert fills[0].side == -1
    assert fills[0].price == 99.0
    assert sim.inventory == -1.0
    assert sim.cash == 99.0


def test_no_fill_when_quotes_do_not_cross() -> None:
    sim = _sim()
    sim.reset(cash=0.0)

    fills = sim.step(
        bid_price=99.0, ask_price=101.0, next_best_bid=99.5, next_best_ask=100.5
    )

    assert fills == []
    assert sim.inventory == 0.0


def test_both_sides_can_fill_in_the_same_step() -> None:
    sim = _sim()
    sim.reset(cash=0.0)

    fills = sim.step(
        bid_price=101.0, ask_price=99.0, next_best_bid=100.0, next_best_ask=100.0
    )

    assert len(fills) == 2
    assert sim.inventory == 0.0  # one buy, one sell -- net flat


def test_quoting_is_blocked_once_max_inventory_reached() -> None:
    sim = _sim(min_unit=1.0, max_inv=1.0)
    sim.reset(cash=0.0)
    sim.inventory = 1.0  # already at max_inventory

    fills = sim.step(
        bid_price=101.0, ask_price=None, next_best_bid=99.0, next_best_ask=100.0
    )

    assert fills == []
    assert sim.inventory == 1.0


def test_close_position_sells_a_long_position_at_next_best_bid() -> None:
    sim = _sim()
    sim.reset(cash=0.0)
    sim.inventory = 2.0

    fills = sim.close_position(next_best_bid=105.0, next_best_ask=106.0)

    assert len(fills) == 1
    assert fills[0].side == -1
    assert fills[0].price == 105.0
    assert fills[0].volume == 2.0
    assert sim.inventory == 0.0
    assert sim.cash == 210.0


def test_close_position_buys_back_a_short_position_at_next_best_ask() -> None:
    sim = _sim()
    sim.reset(cash=0.0)
    sim.inventory = -2.0

    fills = sim.close_position(next_best_bid=105.0, next_best_ask=106.0)

    assert len(fills) == 1
    assert fills[0].side == 1
    assert fills[0].price == 106.0
    assert sim.inventory == 0.0
    assert sim.cash == -212.0


def test_close_position_is_a_noop_when_flat() -> None:
    sim = _sim()
    sim.reset(cash=0.0)

    fills = sim.close_position(next_best_bid=105.0, next_best_ask=106.0)

    assert fills == []
