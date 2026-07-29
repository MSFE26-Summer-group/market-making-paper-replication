"""Tests for paper_replication.features.lob_state."""

import numpy as np
import pytest

from paper_replication.features.lob_state import (
    level_column_names,
    normalize_lob_state,
    rolling_windows,
)


def test_level_column_names_order_and_count() -> None:
    cols = level_column_names("BTCUSDT", n_levels=2)
    assert cols == [
        "BTCUSDT.ask_price_1",
        "BTCUSDT.ask_size_1",
        "BTCUSDT.bid_price_1",
        "BTCUSDT.bid_size_1",
        "BTCUSDT.ask_price_2",
        "BTCUSDT.ask_size_2",
        "BTCUSDT.bid_price_2",
        "BTCUSDT.bid_size_2",
    ]


def test_rolling_windows_shape_and_content() -> None:
    matrix = np.arange(15.0).reshape(5, 3)
    windows = rolling_windows(matrix, window_T=3)

    assert windows.shape == (3, 3, 3)
    np.testing.assert_array_equal(windows[0], matrix[0:3])
    np.testing.assert_array_equal(windows[2], matrix[2:5])


def test_rolling_windows_raises_when_too_short() -> None:
    matrix = np.zeros((2, 3))
    with pytest.raises(ValueError):
        rolling_windows(matrix, window_T=3)


def test_normalize_lob_state_price_and_volume_stats() -> None:
    # n_levels=1 -> columns [ask_price, ask_size, bid_price, bid_size].
    window = np.array(
        [
            [10.0, 1.0, 9.0, 2.0],
            [20.0, 3.0, 19.0, 1.0],
            [30.0, 5.0, 29.0, 4.0],
        ]
    )
    windows = window[np.newaxis, :, :]  # (1, T=3, F=4)

    normed = normalize_lob_state(windows, n_levels=1)

    price_mask = np.array([True, False, True, False])
    prices = normed[0][:, price_mask]
    volumes = normed[0][:, ~price_mask]

    np.testing.assert_allclose(prices.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(prices.std(axis=0), 1.0, atol=1e-10)
    np.testing.assert_allclose(volumes.max(axis=0), 1.0, atol=1e-10)


def test_normalize_lob_state_wrong_feature_count_raises() -> None:
    windows = np.zeros((1, 3, 5))
    with pytest.raises(ValueError):
        normalize_lob_state(windows, n_levels=1)
