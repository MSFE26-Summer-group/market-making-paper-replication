"""Tests for paper_replication.rl.action_space."""

import numpy as np
import pytest

from paper_replication.rl.action_space import ContinuousActionSpace, DiscreteActionSpace


def test_continuous_decode_zero_action_is_centered_zero_spread() -> None:
    space = ContinuousActionSpace(max_bias=0.05, max_spread=0.1)

    quote = space.decode(np.array([0.0, 0.0]), mid_price=100.0, inventory=0.0)

    assert quote.bid_price == pytest.approx(100.0)
    assert quote.ask_price == pytest.approx(100.0)
    assert not quote.close_position


def test_continuous_decode_full_action_uses_max_bias_and_spread() -> None:
    space = ContinuousActionSpace(max_bias=0.05, max_spread=0.1)

    # Positive inventory -> reservation price biased down (Eq. 9).
    quote = space.decode(np.array([1.0, 1.0]), mid_price=100.0, inventory=5.0)

    reservation_price = 100.0 - 0.05
    assert quote.bid_price == pytest.approx(reservation_price - 0.05)
    assert quote.ask_price == pytest.approx(reservation_price + 0.05)


def test_continuous_decode_negative_inventory_biases_price_up() -> None:
    space = ContinuousActionSpace(max_bias=0.05, max_spread=0.1)

    quote = space.decode(np.array([1.0, 0.0]), mid_price=100.0, inventory=-5.0)

    assert quote.bid_price == pytest.approx(100.05)
    assert quote.ask_price == pytest.approx(100.05)


def test_continuous_decode_clips_out_of_range_actions() -> None:
    space = ContinuousActionSpace(max_bias=0.05, max_spread=0.1)

    quote = space.decode(np.array([2.0, -1.0]), mid_price=100.0, inventory=0.0)

    assert quote.bid_price == pytest.approx(100.0)
    assert quote.ask_price == pytest.approx(100.0)


def test_discrete_last_action_closes_position() -> None:
    space = DiscreteActionSpace(max_bias=0.05, max_spread=0.1)

    quote = space.decode(7, mid_price=100.0, inventory=1.0)

    assert quote.close_position
    assert quote.bid_price is None
    assert quote.ask_price is None


def test_discrete_quoting_actions_bracket_mid_price() -> None:
    space = DiscreteActionSpace(max_bias=0.05, max_spread=0.1)

    for action in range(7):
        quote = space.decode(action, mid_price=100.0, inventory=0.0)
        assert quote.bid_price is not None and quote.ask_price is not None
        assert quote.bid_price <= 100.0 <= quote.ask_price
        assert not quote.close_position


def test_discrete_rejects_out_of_range_action() -> None:
    space = DiscreteActionSpace(max_bias=0.05, max_spread=0.1)

    with pytest.raises(ValueError, match="action must be in"):
        space.decode(8, mid_price=100.0, inventory=0.0)
