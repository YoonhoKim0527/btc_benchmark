"""Milestone 3 tests: metrics (graceful zero-trade/all-cash, no div-by-zero)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_benchmark.backtest.metrics import compute_metrics  # noqa: E402


def test_all_zero_returns_graceful():
    m = compute_metrics(np.zeros(100), positions=np.zeros(100), gross_returns=np.zeros(100),
                        costs=np.zeros(100), trades=pd.DataFrame())
    assert m["total_net_return"] == 0.0
    assert np.isnan(m["sharpe"])           # zero volatility -> NaN, not inf
    assert m["n_trades"] == 0
    assert "sharpe NaN" in m["notes"]


def test_no_division_by_zero_on_empty():
    m = compute_metrics(np.array([]))
    assert m["n_periods"] == 0


def test_known_positive_series():
    net = np.full(8760, 0.0001)  # constant tiny positive return for a year (hourly)
    m = compute_metrics(net, periods_per_year=8760)
    assert m["total_net_return"] > 0
    assert m["annualized_return"] > 0
    assert np.isnan(m["sharpe"])  # zero variance -> undefined Sharpe (correctly NaN)


def test_sharpe_finite_with_variance():
    rng = np.random.default_rng(0)
    net = rng.normal(0.0002, 0.01, 8760)
    m = compute_metrics(net, periods_per_year=8760)
    assert np.isfinite(m["sharpe"])
    assert np.isfinite(m["annualized_volatility"])


def test_max_drawdown_sign():
    net = np.array([0.1, -0.5, 0.1])  # big drop
    m = compute_metrics(net)
    assert m["max_drawdown"] < 0


def test_calmar_not_inf_on_arc_overflow():
    # pathologically short/extreme series -> ARC overflows to inf; Calmar must be NaN, not inf
    net = np.array([99.0, -0.5, 0.1])
    m = compute_metrics(net)
    assert not np.isinf(m["calmar"])
