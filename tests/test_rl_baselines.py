"""Tests for paper_replication.rl.baselines."""

import numpy as np

from paper_replication.rl.baselines import (
    AvellanedaStoikovPolicy,
    FixedSpreadPolicy,
    RandomPolicy,
)
from paper_replication.rl.env import Observation


def _obs(inventory_norm: float = 0.0, elapsed_frac: float = 0.0) -> Observation:
    return Observation(
        lob_state=np.zeros((1, 1)),
        dynamic_state=np.array([0.001]),
        agent_state=np.array([inventory_norm, elapsed_frac]),
    )


def test_random_policy_returns_values_in_unit_square() -> None:
    policy = RandomPolicy(seed=0)

    for row in range(20):
        action = policy(_obs(), row)
        assert action.shape == (2,)
        assert np.all(action >= 0.0) and np.all(action <= 1.0)


def test_random_policy_is_reproducible_with_same_seed() -> None:
    a = RandomPolicy(seed=1)(_obs(), 0)
    b = RandomPolicy(seed=1)(_obs(), 0)

    np.testing.assert_array_equal(a, b)


def test_fixed_spread_policy_is_centered_and_constant() -> None:
    policy = FixedSpreadPolicy(spread_frac=0.5)

    action0 = policy(_obs(), 0)
    action5 = policy(_obs(elapsed_frac=0.9), 5)

    np.testing.assert_array_equal(action0, [0.0, 0.5])
    np.testing.assert_array_equal(action5, [0.0, 0.5])


def test_avellaneda_stoikov_policy_returns_valid_action() -> None:
    policy = AvellanedaStoikovPolicy(
        mid_price=np.array([100.0, 100.0, 100.0]),
        max_inventory=5.0,
        max_bias=0.05,
        max_spread=0.1,
        rv_feature_index=0,
    )

    action = policy(_obs(inventory_norm=0.2, elapsed_frac=0.5), 1)

    assert action.shape == (2,)
    assert 0.0 <= action[0] <= 1.0
    assert 0.0 <= action[1] <= 1.0


def test_avellaneda_stoikov_policy_zero_bias_when_flat() -> None:
    policy = AvellanedaStoikovPolicy(
        mid_price=np.array([100.0, 100.0]),
        max_inventory=5.0,
        max_bias=0.05,
        max_spread=0.1,
        rv_feature_index=0,
    )

    action = policy(_obs(inventory_norm=0.0, elapsed_frac=0.0), 0)

    assert action[0] == 0.0  # zero inventory -> zero bias magnitude


def test_avellaneda_stoikov_policy_increases_bias_with_inventory() -> None:
    policy = AvellanedaStoikovPolicy(
        mid_price=np.array([100.0, 100.0]),
        max_inventory=5.0,
        max_bias=0.05,
        max_spread=0.1,
        rv_feature_index=0,
        gamma=0.1,
        kappa=1.5,
    )

    small = policy(_obs(inventory_norm=0.1, elapsed_frac=0.0), 0)
    large = policy(_obs(inventory_norm=1.0, elapsed_frac=0.0), 0)

    assert large[0] >= small[0]
