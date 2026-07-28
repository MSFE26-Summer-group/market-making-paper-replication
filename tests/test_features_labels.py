"""Tests for paper_replication.features.labels."""

import numpy as np
import pytest

from paper_replication.features.labels import (
    calibrate_alpha,
    compute_l_t,
    direction_labels,
)


def test_compute_l_t_matches_hand_calculation() -> None:
    mid_price = np.array([100.0, 110.0, 90.0, 90.0])
    l_t = compute_l_t(mid_price, k=1)

    assert np.isnan(l_t[3])
    np.testing.assert_allclose(l_t[0], (110.0 - 100.0) / 100.0)
    np.testing.assert_allclose(l_t[1], (90.0 - 110.0) / 110.0)
    np.testing.assert_allclose(l_t[2], (90.0 - 90.0) / 90.0)


def test_compute_l_t_multi_point_window() -> None:
    # past_mean(1) = mean(p[0:2]) = 105; future_mean(1) = mean(p[2:4]) = 89.5
    mid_price = np.array([100.0, 110.0, 90.0, 89.0, 80.0])
    l_t = compute_l_t(mid_price, k=2)
    assert np.isnan(l_t[0])
    np.testing.assert_allclose(l_t[1], (89.5 - 105.0) / 105.0)


def test_direction_labels_thresholds() -> None:
    l_t = np.array([0.02, -0.02, 0.005, np.nan])
    labels = direction_labels(l_t, alpha=0.01)
    np.testing.assert_allclose(labels[:3], [1.0, -1.0, 0.0])
    assert np.isnan(labels[3])


def test_calibrate_alpha_yields_target_class_balance() -> None:
    rng = np.random.default_rng(0)
    l_t = rng.normal(scale=0.05, size=20000)
    target = 1.0 / 3.0

    alpha = calibrate_alpha(l_t, target_balance=target)
    labels = direction_labels(l_t, alpha)

    up_frac = np.mean(labels == 1.0)
    down_frac = np.mean(labels == -1.0)
    np.testing.assert_allclose(up_frac, target, atol=0.01)
    np.testing.assert_allclose(down_frac, target, atol=0.01)


def test_calibrate_alpha_rejects_bad_target_balance() -> None:
    with pytest.raises(ValueError):
        calibrate_alpha(np.array([0.1, 0.2]), target_balance=0.6)
