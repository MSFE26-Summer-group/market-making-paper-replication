"""Tests for paper_replication.rl.config."""

from paper_replication.rl.config import RLConfig


def test_max_inventory_is_omega_times_minimum_trade_unit() -> None:
    config = RLConfig(minimum_trade_unit=0.01, omega=5.0)

    assert config.max_inventory == 0.05


def test_defaults_match_documented_paper_values() -> None:
    config = RLConfig()

    assert config.omega == 10.0
    assert config.eta == 0.5
    assert config.zeta == 0.01
    assert config.max_bias == 0.05
    assert config.max_spread == 0.1
