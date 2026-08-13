"""Tests for paper_replication.training.config."""

import pytest

from paper_replication.training.config import TrainConfig


def test_default_device_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "paper_replication.training.config.torch.cuda.is_available", lambda: True
    )
    assert TrainConfig().device == "cuda"


def test_default_device_falls_back_to_cpu_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "paper_replication.training.config.torch.cuda.is_available", lambda: False
    )
    assert TrainConfig().device == "cpu"


def test_explicit_device_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "paper_replication.training.config.torch.cuda.is_available", lambda: True
    )
    assert TrainConfig(device="cpu").device == "cpu"


def test_default_patience_matches_documented_value() -> None:
    """Regression test: patience silently drifted to 10 while every docstring,
    notebook, and experiment writeup still said 5 -- nothing caught it until
    a notebook re-run produced quietly different numbers. Pins the default
    so that can't happen again without a loud test failure.
    """
    assert TrainConfig().patience == 5
