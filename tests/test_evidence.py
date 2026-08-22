"""Unit tests for the public evidence-receipt helpers."""

from __future__ import annotations

import pytest

from pico_engine.evidence import _percentile, _summary


def test_percentile_interpolates_sorted_samples():
    values = [50.0, 10.0, 30.0, 20.0, 40.0]
    assert _percentile(values, 0.0) == 10.0
    assert _percentile(values, 0.5) == 30.0
    assert _percentile(values, 1.0) == 50.0


def test_summary_reports_distribution_not_only_one_point():
    summary = _summary([10.0, 20.0, 30.0, 40.0, 50.0])
    assert summary == {
        "count": 5,
        "median": 30.0,
        "p10": 14.0,
        "p90": 46.0,
        "minimum": 10.0,
        "maximum": 50.0,
    }


def test_percentile_rejects_empty_samples():
    with pytest.raises(ValueError, match="at least one"):
        _percentile([], 0.5)
