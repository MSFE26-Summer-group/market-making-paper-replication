"""Tests for paper_replication.rl.state."""

import numpy as np

from paper_replication.rl.state import agent_state_vector


def test_agent_state_vector_normalizes_inventory_and_keeps_elapsed_frac() -> None:
    vec = agent_state_vector(inventory=5.0, max_inventory=10.0, elapsed_frac=0.25)

    np.testing.assert_allclose(vec, [0.5, 0.25])


def test_agent_state_vector_handles_zero_max_inventory() -> None:
    vec = agent_state_vector(inventory=0.0, max_inventory=0.0, elapsed_frac=0.0)

    np.testing.assert_allclose(vec, [0.0, 0.0])


def test_agent_state_vector_dtype_is_float64() -> None:
    vec = agent_state_vector(inventory=1.0, max_inventory=1.0, elapsed_frac=1.0)

    assert vec.dtype == np.float64
    assert vec.shape == (2,)
