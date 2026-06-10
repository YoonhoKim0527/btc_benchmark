"""Milestone 3 tests: cost model + turnover convention."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_benchmark.backtest.cost_model import CostConfig, cost_series, turnover_series  # noqa: E402


def test_turnover_table():
    # p_{-1}=0; turnover_t = |p_t - p_{t-1}|
    assert list(turnover_series([1, 0])) == [1, 1]      # cash->long=1, long->cash=1
    assert list(turnover_series([1, -1])) == [1, 2]     # long->short=2
    assert list(turnover_series([-1, 1])) == [1, 2]     # short->long=2
    assert list(turnover_series([0, 0])) == [0, 0]      # cash->cash=0
    assert list(turnover_series([1, 1])) == [1, 0]      # long->long=0
    assert list(turnover_series([-1, -1])) == [1, 0]    # short->short=0


def test_all_in_cost_bps_and_scaling():
    cfg = CostConfig(fee_bps=10, slippage_bps=2, safety_buffer_bps=3)
    assert cfg.all_in_cost_bps == 15
    assert cfg.scaled(2.0).all_in_cost_bps == 30
    assert cfg.scaled(0.0).all_in_cost_bps == 0


def test_cost_series_math():
    cfg = CostConfig(fee_bps=10, slippage_bps=0, safety_buffer_bps=0)  # 10 bps = 0.001
    cost = cost_series([1, -1, -1], cfg)  # turnover [1,2,0]
    assert cost == pytest.approx([0.0001 * 1 * 10, 0.0001 * 2 * 10, 0.0])  # 0.001, 0.002, 0
