"""End-to-end test for paper_replication.features.dataset on a synthetic LOB file."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_replication.features.config import FeatureConfig
from paper_replication.features.dataset import (
    build_feature_dataset,
    chronological_split,
)


@pytest.fixture
def lob_parquet(tmp_path: Path) -> str:
    rng = np.random.default_rng(42)
    n = 30
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
        horizon_k=2,
        osi_windows_seconds=(50,),
        rv_windows_seconds=(50,),
        rsi_windows_seconds=(50,),
    )


def test_build_feature_dataset_shapes(lob_parquet: str) -> None:
    dataset = build_feature_dataset(lob_parquet, _small_config())

    assert dataset.lob_state.ndim == 3
    assert dataset.lob_state.shape[1:] == (3, 4)  # window_T=3, 4*n_levels=4
    assert dataset.dynamic_state.shape[0] == len(dataset)
    assert dataset.dynamic_state.shape[1] == len(dataset.dynamic_feature_names)
    assert set(np.unique(dataset.labels)).issubset({-1, 0, 1})
    assert len(dataset.timestamps) == len(dataset)
    assert np.all(np.diff(dataset.timestamps) > 0)  # still time-ordered


def test_chronological_split_partitions_without_overlap(lob_parquet: str) -> None:
    dataset = build_feature_dataset(lob_parquet, _small_config())
    splits = chronological_split(dataset, test_frac=0.3, val_frac_of_train=0.25)

    total = sum(len(s) for s in splits.values())
    assert total == len(dataset)

    # Chronological: train ends before val starts before test starts.
    if len(splits["train"]) and len(splits["val"]):
        assert splits["train"].timestamps[-1] <= splits["val"].timestamps[0]
    if len(splits["val"]) and len(splits["test"]):
        assert splits["val"].timestamps[-1] <= splits["test"].timestamps[0]
