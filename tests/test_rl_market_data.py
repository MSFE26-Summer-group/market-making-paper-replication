"""End-to-end test for paper_replication.rl.market_data on a synthetic LOB file."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_replication.features.config import FeatureConfig
from paper_replication.rl.market_data import build_market_dataset, chronological_split


@pytest.fixture
def lob_parquet(tmp_path: Path) -> str:
    rng = np.random.default_rng(7)
    n = 40
    mid = 100.0 + np.cumsum(rng.normal(scale=0.1, size=n))
    df = pd.DataFrame(
        {
            "timestamp": np.arange(n, dtype=float) * 10.0,
            "BTCUSDT.ask_price_1": mid + 0.5,
            "BTCUSDT.ask_size_1": rng.uniform(0.1, 1.0, n),
            "BTCUSDT.bid_price_1": mid - 0.5,
            "BTCUSDT.bid_size_1": rng.uniform(0.1, 1.0, n),
            "mid_price": mid,
            "net_trade_sign": rng.integers(-5, 6, n),
            "trade_count": rng.integers(1, 10, n),
            "ema_ofi": rng.normal(scale=0.1, size=n),
        }
    )
    path = tmp_path / "lob.parquet"
    df.to_parquet(path)
    return str(path)


def _small_config() -> FeatureConfig:
    return FeatureConfig(
        n_levels=1,
        window_T=3,
        osi_windows_seconds=(50,),
        rv_windows_seconds=(50,),
        rsi_windows_seconds=(50,),
    )


def test_build_market_dataset_shapes(lob_parquet: str) -> None:
    dataset = build_market_dataset(lob_parquet, _small_config())

    assert dataset.lob_state.ndim == 3
    assert dataset.lob_state.shape[1:] == (3, 4)  # window_T=3, 4*n_levels=4
    assert dataset.dynamic_state.shape[0] == len(dataset)
    assert dataset.dynamic_state.shape[1] == len(dataset.dynamic_feature_names)
    assert dataset.best_ask.shape == (len(dataset),)
    assert dataset.best_bid.shape == (len(dataset),)
    assert dataset.mid_price.shape == (len(dataset),)
    assert len(dataset.timestamps) == len(dataset)


def test_build_market_dataset_best_prices_bracket_mid_price(lob_parquet: str) -> None:
    dataset = build_market_dataset(lob_parquet, _small_config())

    assert np.all(dataset.best_bid < dataset.mid_price)
    assert np.all(dataset.mid_price < dataset.best_ask)


def test_build_market_dataset_is_time_ordered(lob_parquet: str) -> None:
    dataset = build_market_dataset(lob_parquet, _small_config())

    assert np.all(np.diff(dataset.timestamps) > 0)


def test_build_market_dataset_drops_rows_before_first_full_window(
    lob_parquet: str,
) -> None:
    config = _small_config()
    dataset = build_market_dataset(lob_parquet, config)

    df = pd.read_parquet(lob_parquet)
    n = len(df)
    assert len(dataset) <= n - (config.window_T - 1)


def test_chronological_split_partitions_without_overlap(lob_parquet: str) -> None:
    dataset = build_market_dataset(lob_parquet, _small_config())

    splits = chronological_split(dataset, test_frac=0.3)

    assert len(splits["train"]) + len(splits["test"]) == len(dataset)
    if len(splits["train"]) and len(splits["test"]):
        assert splits["train"].timestamps[-1] <= splits["test"].timestamps[0]


def test_chronological_split_test_frac_controls_the_split_point(
    lob_parquet: str,
) -> None:
    dataset = build_market_dataset(lob_parquet, _small_config())

    splits = chronological_split(dataset, test_frac=0.5)

    assert abs(len(splits["test"]) - len(dataset) // 2) <= 1
