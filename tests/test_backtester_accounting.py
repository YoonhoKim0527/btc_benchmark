"""Milestone 3 tests: backtester accounting on tiny synthetic series (hand-computable).

Mode A (close_to_close_reference) convention: p_t (decided at bar t) earns close_t -> close_{t+1};
entering p_t costs |p_t - p_{t-1}| charged in that period.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_benchmark.backtest.backtester import run_backtest  # noqa: E402
from btc_benchmark.backtest.cost_model import CostConfig, turnover_series  # noqa: E402


def candles(closes, opens=None, imputed=None, start="2020-01-01"):
    n = len(closes)
    ts = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    closes = np.asarray(closes, dtype="float64")
    opens = np.asarray(opens, dtype="float64") if opens is not None else closes.copy()
    return pd.DataFrame({
        "timestamp_open": ts,
        "timestamp_close": ts + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        "open": opens, "high": np.maximum(opens, closes) + 1.0, "low": np.minimum(opens, closes) - 1.0,
        "close": closes,
        "is_imputed": (imputed if imputed is not None else [False] * n),
    })


C10 = CostConfig(fee_bps=10)   # 0.001 per unit turnover
C0 = CostConfig(fee_bps=0)


# 1
def test_all_cash_zero_return_zero_cost():
    res = run_backtest(candles([100, 101, 102, 103]), [0, 0, 0, 0], C10)
    assert np.allclose(res.net_returns, 0.0)
    assert res.costs.sum() == 0.0
    assert res.equity[-1] == pytest.approx(1.0)
    assert len(res.trades) == 0


# 2
def test_buy_and_hold_long_one_entry_cost():
    res = run_backtest(candles([100, 110, 121]), [1, 1, 1], C10)
    assert res.metrics()["total_gross_return"] == pytest.approx(0.21)
    assert res.costs.sum() == pytest.approx(0.001)  # one entry from cash, no further turnover
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["entry_price"] == 100
    assert res.trades.iloc[0]["exit_price"] == 121


# 3
def test_long_to_cash_turnover_one():
    assert turnover_series([1, 0])[1] == 1
    res = run_backtest(candles([100, 100, 100]), [1, 0, 0], C10)
    assert res.costs[0] == pytest.approx(0.001)  # cash->long
    assert res.costs[1] == pytest.approx(0.001)  # long->cash, turnover 1


# 4
def test_long_to_short_turnover_two():
    assert turnover_series([1, -1])[1] == 2
    res = run_backtest(candles([100, 100, 100]), [1, -1, -1], C10)
    assert res.costs[1] == pytest.approx(0.002)  # long->short, turnover 2


# 5
def test_short_return_sign():
    res = run_backtest(candles([100, 90]), [-1, -1], C0)
    assert res.gross_returns[0] == pytest.approx(0.1)  # short gains 10% when price falls 10%


# 6
def test_cost_reduces_net():
    res = run_backtest(candles([100, 110]), [1, 1], C10)
    assert res.net_returns[0] == pytest.approx(res.gross_returns[0] - 0.001)


# 7
def test_zero_cost_equals_gross():
    res = run_backtest(candles([100, 110, 121]), [1, -1, 1], C10.scaled(0.0))
    assert np.allclose(res.net_returns, res.gross_returns)
    assert res.costs.sum() == 0.0


# 8
def test_2x_cost_doubles_cost():
    c, pos = candles([100, 110, 121]), [1, -1, 1]
    r1 = run_backtest(c, pos, C10.scaled(1.0))
    r2 = run_backtest(c, pos, C10.scaled(2.0))
    assert r2.costs.sum() == pytest.approx(2.0 * r1.costs.sum())


# 9
def test_flat_prices_high_turnover_loses_to_cost():
    res = run_backtest(candles([100] * 11), [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], C10)
    assert res.equity[-1] < 1.0
    assert res.metrics()["total_net_return"] < 0


# 10
def test_trade_log_times_and_prices():
    c = candles([100, 101, 102, 103, 104])
    res = run_backtest(c, [0, 1, 1, 0, 0], C0)
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["side"] == "long"
    assert tr["entry_price"] == 101 and tr["exit_price"] == 103
    assert tr["holding_period_bars"] == 2
    assert tr["exit_reason"] == "signal"
    assert pd.Timestamp(tr["entry_time"]) == c["timestamp_close"].iloc[1]
    assert pd.Timestamp(tr["exit_time"]) == c["timestamp_close"].iloc[3]


# 11
def test_imputed_candle_flagged_in_trade_log():
    res = run_backtest(candles([100, 101, 102], imputed=[False, True, False]), [1, 1, 1], C0)
    assert bool(res.trades.iloc[0]["touched_imputed_candle"]) is True


# 12
def test_funding_applies_only_at_event():
    c = candles([100, 100, 100])  # flat -> gross 0, isolate funding
    funding = pd.DataFrame({"event_time": pd.to_datetime(["2020-01-01 01:00:00"], utc=True),
                            "funding_rate": [0.01]})
    res = run_backtest(c, [1, 1, 1], CostConfig(fee_bps=0, funding_enabled=True), funding=funding)
    assert res.funding[0] == pytest.approx(-0.01)  # event in period 0; long pays when rate>0
    assert res.funding[1] == pytest.approx(0.0)    # no event in period 1


# 13
def test_funding_disabled_runs_clean():
    c = candles([100, 100, 100])
    funding = pd.DataFrame({"event_time": pd.to_datetime(["2020-01-01 01:00:00"], utc=True),
                            "funding_rate": [0.01]})
    res = run_backtest(c, [1, 1, 1], CostConfig(fee_bps=0, funding_enabled=False), funding=funding)
    assert np.allclose(res.funding, 0.0)


# 14
def test_no_off_by_one_close_to_close():
    res = run_backtest(candles([100, 110, 121]), [1, 0, 0], C0)
    assert res.gross_returns[0] == pytest.approx(0.1)  # p0 earns close0->close1
    assert res.gross_returns[1] == pytest.approx(0.0)  # p1=0 earns nothing


# --- Mode B + guards -----------------------------------------------------------
def test_mode_b_open_to_open():
    c = candles(closes=[100, 100, 100, 100], opens=[100, 110, 121, 133.1])
    res = run_backtest(c, [1, 1, 1, 1], C0, execution_mode="next_open_conservative")
    assert res.gross_returns[0] == pytest.approx(0.1)  # open2/open1 - 1 = 121/110 - 1
    assert res.execution_mode == "next_open_conservative"


def test_mode_b_requires_open_column():
    c = candles([100, 101, 102]).drop(columns=["open"])
    with pytest.raises(KeyError):
        run_backtest(c, [1, 1, 1], C0, execution_mode="next_open_conservative")


def test_positions_length_mismatch_raises():
    with pytest.raises(ValueError):
        run_backtest(candles([100, 101, 102]), [1, 1], C0)


def test_backtester_does_not_mutate_input():
    c = candles([100, 110, 121])
    before = c.copy()
    run_backtest(c, [1, 1, 1], C10)
    pd.testing.assert_frame_equal(c, before)


# --- input guards added after code review --------------------------------------
def test_unsorted_timestamps_raise():
    c = candles([100, 101, 102, 103]).iloc[[0, 2, 1, 3]].reset_index(drop=True)
    with pytest.raises(ValueError):
        run_backtest(c, [0, 0, 0, 0], C0)


def test_duplicate_timestamps_raise():
    c = candles([100, 101, 102])
    c.loc[2, "timestamp_open"] = c.loc[1, "timestamp_open"]
    with pytest.raises(ValueError):
        run_backtest(c, [0, 0, 0], C0)


def test_nan_close_hard_fails():
    c = candles([100, 101, 102])
    c.loc[1, "close"] = np.nan
    with pytest.raises(ValueError):
        run_backtest(c, [1, 1, 1], C0)


def test_mode_b_needs_three_bars():
    with pytest.raises(ValueError):
        run_backtest(candles([100, 101]), [1, 1], C0, execution_mode="next_open_conservative")


def test_fractional_positions_accounting():
    res = run_backtest(candles([100, 110, 121]), [0.5, 0.5, -0.25], C10)
    assert res.gross_returns[0] == pytest.approx(0.5 * 0.1)   # half-size long earns half the move
    assert res.turnover[0] == pytest.approx(0.5)              # entering 0.5 from cash


def test_funding_short_receives_when_rate_positive():
    c = candles([100, 100, 100])
    f = pd.DataFrame({"event_time": pd.to_datetime(["2020-01-01 01:00:00"], utc=True), "funding_rate": [0.01]})
    res = run_backtest(c, [-1, -1, -1], CostConfig(fee_bps=0, funding_enabled=True), funding=f)
    assert res.funding[0] == pytest.approx(0.01)  # short receives when rate > 0


def test_funding_boundary_right_inclusive_left_exclusive():
    c = candles([100, 100, 100])
    tc = c["timestamp_close"]
    # event exactly at close_1 (right edge of period 0) -> attached to period 0
    f_right = pd.DataFrame({"event_time": [tc.iloc[1]], "funding_rate": [0.01]})
    r = run_backtest(c, [1, 1, 1], CostConfig(fee_bps=0, funding_enabled=True), funding=f_right)
    assert r.funding[0] == pytest.approx(-0.01) and r.funding[1] == pytest.approx(0.0)
    # event exactly at close_0 (left edge of period 0) -> NOT attached (no earlier period)
    f_left = pd.DataFrame({"event_time": [tc.iloc[0]], "funding_rate": [0.01]})
    r2 = run_backtest(c, [1, 1, 1], CostConfig(fee_bps=0, funding_enabled=True), funding=f_left)
    assert np.allclose(r2.funding, 0.0)
