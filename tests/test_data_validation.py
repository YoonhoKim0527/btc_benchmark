"""Milestone 1 tests: data validation, missing-candle handling, schema guards.

These run fully offline on synthetic frames (no network, no parquet IO required).
Run: pytest tests/test_data_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# make `src` importable when running pytest from the project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_benchmark.data import download_binance as dl  # noqa: E402
from btc_benchmark.data.impute import build_grid  # noqa: E402
from btc_benchmark.data.schema import (  # noqa: E402
    DataQualityFlag,
    FLAT_BAR_METHOD,
    PROCESSED_COLUMNS,
    validate_history_request,
)
from btc_benchmark.data.validate_data import validate  # noqa: E402
from btc_benchmark.utils.time import (  # noqa: E402
    detect_epoch_unit,
    epoch_series_to_utc,
    epoch_to_utc,
    interval_timedelta,
)

DOWNLOADED_AT = "2026-01-01T00:00:00+00:00"  # fixed -> deterministic / idempotent


def make_clean(n: int = 48, interval_h: int = 1, start="2020-01-01") -> pd.DataFrame:
    """A clean, gap-free hourly canonical frame."""
    ts = pd.date_range(start, periods=n, freq=f"{interval_h}h", tz="UTC")
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "timestamp_open": ts,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 10.0),
            "quote_volume": np.full(n, 1000.0),
            "number_of_trades": np.full(n, 5),
            "taker_buy_base_volume": np.full(n, 5.0),
            "taker_buy_quote_volume": np.full(n, 500.0),
        }
    )


# --- validator -----------------------------------------------------------------
def test_clean_passes():
    rep = validate(make_clean(), interval="1h", market_type="futures_um", symbol="BTCUSDT")
    assert rep["passed_hard_checks"] is True
    assert rep["n_missing_bars"] == 0
    assert rep["n_duplicate_timestamps"] == 0
    assert rep["n_invalid_ohlc"] == 0
    assert rep["n_nonpositive_price"] == 0


def test_duplicate_timestamp_flagged():
    df = make_clean()
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True).sort_values("timestamp_open")
    rep = validate(df, interval="1h")
    assert rep["n_duplicate_timestamps"] == 1
    assert rep["passed_hard_checks"] is False


def test_non_monotonic_flagged():
    df = make_clean()
    df = df.iloc[::-1].reset_index(drop=True)  # reversed -> every step decreases
    rep = validate(df, interval="1h")
    assert rep["n_non_monotonic"] > 0
    assert rep["passed_hard_checks"] is False


def test_invalid_ohlc_flagged():
    df = make_clean()
    df.loc[5, "high"] = df.loc[5, "low"] - 1.0  # high < low
    rep = validate(df, interval="1h")
    assert rep["n_invalid_ohlc"] >= 1
    assert 5 in rep["invalid_ohlc_sample"]
    assert rep["passed_hard_checks"] is False


def test_nonpositive_price_flagged():
    df = make_clean()
    df.loc[7, "close"] = 0.0
    rep = validate(df, interval="1h")
    assert rep["n_nonpositive_price"] >= 1
    assert rep["passed_hard_checks"] is False


def test_negative_volume_flagged():
    df = make_clean()
    df.loc[3, "volume"] = -1.0
    rep = validate(df, interval="1h")
    assert rep["n_negative_volume"] == 1
    assert rep["passed_hard_checks"] is False


def test_missing_bar_detected():
    df = make_clean(n=48)
    df = df.drop(index=[20, 21]).reset_index(drop=True)  # 2-hour gap
    rep = validate(df, interval="1h")
    assert rep["n_missing_bars"] == 2
    assert rep["n_irregular_gaps"] == 1


def test_suspicious_return_flagged():
    df = make_clean()
    df.loc[12, "close"] = df.loc[11, "close"] * 2.0  # ~69% jump
    rep = validate(df, interval="1h", suspicious_logret=0.25)
    assert rep["n_suspicious_returns"] >= 1


# --- imputation ----------------------------------------------------------------
def test_flat_bar_fill_creates_one_imputed_bar():
    df = make_clean(n=24)
    prev_close = float(df.loc[9, "close"])  # bar at index 10 will be removed
    df = df.drop(index=[10]).reset_index(drop=True)
    out, rep = build_grid(
        df, interval="1h", symbol="BTCUSDT", market_type="futures_um",
        downloaded_at=DOWNLOADED_AT, policy="flat_bar_fill",
    )
    assert rep["n_imputed_bars"] == 1
    assert rep["n_total_bars"] == 24  # grid fully reconstructed
    imp = out.loc[out["is_imputed"]]
    assert len(imp) == 1
    row = imp.iloc[0]
    assert row["open"] == row["high"] == row["low"] == row["close"] == prev_close
    assert row["volume"] == 0
    assert row["imputation_method"] == FLAT_BAR_METHOD
    assert row["data_quality_flag"] == DataQualityFlag.MISSING_FILLED
    assert list(out.columns) == PROCESSED_COLUMNS


def test_consecutive_gaps_all_flat_from_prev_close():
    df = make_clean(n=24)
    prev_close = float(df.loc[9, "close"])
    df = df.drop(index=[10, 11, 12]).reset_index(drop=True)
    out, rep = build_grid(
        df, interval="1h", symbol="BTCUSDT", market_type="futures_um",
        downloaded_at=DOWNLOADED_AT,
    )
    assert rep["n_imputed_bars"] == 3
    assert (out.loc[out["is_imputed"], "close"] == prev_close).all()


def test_leading_edge_gap_dropped_not_filled():
    df = make_clean(n=24)
    # remove the FIRST two bars -> no previous close exists to carry forward
    kept = df.iloc[2:].reset_index(drop=True)
    out, rep = build_grid(
        kept, interval="1h", symbol="BTCUSDT", market_type="futures_um",
        downloaded_at=DOWNLOADED_AT,
    )
    # grid starts at the first REAL bar; nothing fabricated before it
    assert rep["n_leading_dropped"] == 0  # we never reindex before the first real bar
    assert out["timestamp_open"].min() == kept["timestamp_open"].min()
    assert bool(out["is_imputed"].iloc[0]) is False


def test_internal_gap_after_offset_start_is_filled():
    df = make_clean(n=24)
    df = df.drop(index=[15]).reset_index(drop=True)
    out, rep = build_grid(
        df, interval="1h", symbol="BTCUSDT", market_type="futures_um",
        downloaded_at=DOWNLOADED_AT,
    )
    assert rep["n_imputed_bars"] == 1


def test_strict_drop_keeps_only_real_bars():
    df = make_clean(n=24).drop(index=[10, 11]).reset_index(drop=True)
    out, rep = build_grid(
        df, interval="1h", symbol="BTCUSDT", market_type="futures_um",
        downloaded_at=DOWNLOADED_AT, policy="strict_drop",
    )
    assert rep["n_imputed_bars"] == 0
    assert rep["n_total_bars"] == 22
    assert not out["is_imputed"].any()


def test_impute_is_idempotent():
    df = make_clean(n=24).drop(index=[10, 13]).reset_index(drop=True)
    out1, _ = build_grid(
        df, interval="1h", symbol="BTCUSDT", market_type="futures_um", downloaded_at=DOWNLOADED_AT,
    )
    out2, _ = build_grid(
        out1, interval="1h", symbol="BTCUSDT", market_type="futures_um", downloaded_at=DOWNLOADED_AT,
    )
    pd.testing.assert_frame_equal(out1, out2)


def test_processed_passes_validator():
    df = make_clean(n=24).drop(index=[10]).reset_index(drop=True)
    out, _ = build_grid(
        df, interval="1h", symbol="BTCUSDT", market_type="futures_um", downloaded_at=DOWNLOADED_AT,
    )
    rep = validate(out, interval="1h", market_type="futures_um", symbol="BTCUSDT")
    assert rep["passed_hard_checks"] is True
    assert rep["n_missing_bars"] == 0
    assert rep["pct_imputed"] is not None and rep["pct_imputed"] > 0


# --- schema / guards -----------------------------------------------------------
def test_futures_pre_launch_request_raises():
    with pytest.raises(ValueError):
        validate_history_request("futures_um", "2017-12-01")


def test_futures_post_launch_request_ok():
    validate_history_request("futures_um", "2019-09-10")  # no raise


def test_spot_early_history_ok():
    validate_history_request("spot", "2017-08-01")  # no raise


# --- timestamp unit detection (ms vs microseconds) -----------------------------
def test_epoch_unit_detection_ms_vs_us():
    # 2020-01-01T00:00:00Z
    ms = 1_577_836_800_000
    us = 1_577_836_800_000_000
    assert detect_epoch_unit(ms) == "ms"
    assert detect_epoch_unit(us) == "us"
    assert epoch_to_utc(ms) == epoch_to_utc(us)  # same instant despite different units


# --- pure URL builders ---------------------------------------------------------
def test_url_builders():
    assert dl.monthly_kline_url("futures_um", "BTCUSDT", "1h", 2020, 1) == (
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2020-01.zip"
    )
    assert dl.monthly_kline_url("spot", "BTCUSDT", "1m", 2017, 8) == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2017-08.zip"
    )
    assert dl.daily_kline_url("futures_um", "BTCUSDT", "15m", pd.Timestamp("2024-03-05")) == (
        "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/15m/BTCUSDT-15m-2024-03-05.zip"
    )


def test_raw_to_canonical_maps_columns_and_parses_time():
    raw = pd.DataFrame(
        [[1_577_836_800_000, 1, 2, 0.5, 1.5, 10, 1_577_840_399_999, 20, 7, 5, 8, 0]],
        columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    canon = dl.raw_to_canonical(raw, symbol="BTCUSDT", market_type="futures_um", interval="1h")
    assert canon["timestamp_open"].iloc[0] == pd.Timestamp("2020-01-01", tz="UTC")
    assert canon["taker_buy_base_volume"].iloc[0] == 5.0
    assert {"open", "high", "low", "close", "volume", "quote_volume"}.issubset(canon.columns)


def test_read_kline_frame_handles_header_row():
    csv = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,"
        "number_of_trades,taker_buy_base,taker_buy_quote,ignore\n"
        "1577836800000,1,2,0.5,1.5,10,1577840399999,20,7,5,8,0\n"
    )
    import io as _io

    frame = dl.read_kline_frame(_io.StringIO(csv))
    assert len(frame) == 1  # header row dropped
    assert float(frame["open"].iloc[0]) == 1.0


# --- hardening tests added after code review -----------------------------------
def test_epoch_mixed_units_in_one_series():
    """ms and us rows mixed in one Series (the 2025-01-01 spot boundary) map correctly."""
    ms = 1_577_836_800_000        # 2020-01-01T00:00:00Z in ms
    us = 1_577_836_800_000_000    # same instant in microseconds
    out = epoch_series_to_utc(pd.Series([ms, us, ms]))
    assert (out == pd.Timestamp("2020-01-01", tz="UTC")).all()


def test_strict_drop_removes_carried_imputed_bars():
    """Feeding flat-filled output into strict_drop must remove the synthetic bars."""
    df = make_clean(n=24).drop(index=[10, 11]).reset_index(drop=True)
    filled, _ = build_grid(
        df, interval="1h", symbol="BTCUSDT", market_type="futures_um",
        downloaded_at=DOWNLOADED_AT, policy="flat_bar_fill",
    )
    assert int(filled["is_imputed"].sum()) == 2
    dropped, rep = build_grid(
        filled, interval="1h", symbol="BTCUSDT", market_type="futures_um",
        downloaded_at=DOWNLOADED_AT, policy="strict_drop",
    )
    assert rep["n_imputed_bars"] == 0
    assert not dropped["is_imputed"].any()
    assert len(dropped) == 22  # the two synthetic bars are gone


def test_validator_flags_nan_ohlc():
    df = make_clean(n=12)
    df.loc[4, "close"] = np.nan
    rep = validate(df, interval="1h")
    assert rep["n_nan_price"] >= 1
    assert rep["passed_hard_checks"] is False


def test_validator_flags_off_grid_timestamp():
    df = make_clean(n=12)
    df.loc[5, "timestamp_open"] = df.loc[5, "timestamp_open"] + pd.Timedelta(minutes=30)
    rep = validate(df, interval="1h")
    assert rep["n_off_grid_timestamps"] >= 1
    assert rep["passed_hard_checks"] is False


def test_impute_drops_off_grid_with_report():
    df = make_clean(n=12)
    extra = df.iloc[[5]].copy()
    extra["timestamp_open"] = extra["timestamp_open"] + pd.Timedelta(minutes=30)
    df2 = pd.concat([df, extra], ignore_index=True)
    out, rep = build_grid(
        df2, interval="1h", symbol="BTCUSDT", market_type="futures_um", downloaded_at=DOWNLOADED_AT,
    )
    assert rep["n_off_grid_dropped"] == 1
    assert (out["timestamp_open"].dt.minute == 0).all()  # the :30 bar is gone


def test_futures_launch_month_boundary():
    validate_history_request("futures_um", "2019-09-01")  # launch month: allowed
    with pytest.raises(ValueError):
        validate_history_request("futures_um", "2019-08-31")  # before launch month: raises


@pytest.mark.parametrize("interval,freq", [("1m", "1min"), ("1d", "1D")])
def test_timestamp_close_convention(interval, freq):
    ts = pd.date_range("2020-01-01", periods=6, freq=freq, tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp_open": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
            "volume": 1.0, "quote_volume": 1.0, "number_of_trades": 1,
            "taker_buy_base_volume": 1.0, "taker_buy_quote_volume": 1.0,
        }
    )
    out, _ = build_grid(
        df, interval=interval, symbol="BTCUSDT", market_type="futures_um", downloaded_at=DOWNLOADED_AT,
    )
    expected = out["timestamp_open"] + interval_timedelta(interval) - pd.Timedelta(milliseconds=1)
    assert (out["timestamp_close"] == expected).all()


def test_number_of_trades_is_nullable_int():
    df = make_clean(n=12).drop(index=[5]).reset_index(drop=True)
    out, _ = build_grid(
        df, interval="1h", symbol="BTCUSDT", market_type="futures_um", downloaded_at=DOWNLOADED_AT,
    )
    assert str(out["number_of_trades"].dtype) == "Int64"
    # imputed bar has 0 trades, real bars keep their integer count
    assert int(out.loc[out["is_imputed"], "number_of_trades"].iloc[0]) == 0


# --- M1.1: expected_start/expected_end edge-gap detection ----------------------
def test_expected_range_detects_missing_first_bar():
    full = make_clean(n=12)  # 00:00 .. 11:00
    data = full.iloc[1:].reset_index(drop=True)  # drop the first bar
    rep0 = validate(data, interval="1h")  # no expected range
    assert rep0["n_missing_bars"] == 0  # prefix gap invisible (bounded by data min)
    assert rep0["prefix_detectable"] is False
    rep = validate(
        data, interval="1h",
        expected_start=full["timestamp_open"].iloc[0],
        expected_end=full["timestamp_open"].iloc[-1],
    )
    assert rep["n_missing_bars"] == 1
    assert rep["n_missing_prefix"] == 1
    assert rep["n_missing_suffix"] == 0
    assert rep["prefix_detectable"] is True


def test_expected_range_detects_missing_last_bar():
    full = make_clean(n=12)
    data = full.iloc[:-1].reset_index(drop=True)  # drop the last bar
    rep0 = validate(data, interval="1h")
    assert rep0["n_missing_bars"] == 0  # suffix gap invisible without expected_end
    rep = validate(
        data, interval="1h",
        expected_start=full["timestamp_open"].iloc[0],
        expected_end=full["timestamp_open"].iloc[-1],
    )
    assert rep["n_missing_bars"] == 1
    assert rep["n_missing_suffix"] == 1
    assert rep["suffix_detectable"] is True


def test_expected_range_detects_missing_middle_bar():
    full = make_clean(n=12)
    data = full.drop(index=[5]).reset_index(drop=True)
    rep = validate(
        data, interval="1h",
        expected_start=full["timestamp_open"].iloc[0],
        expected_end=full["timestamp_open"].iloc[-1],
    )
    assert rep["n_missing_bars"] == 1
    assert rep["n_missing_prefix"] == 0
    assert rep["n_missing_suffix"] == 0


def test_expected_range_no_missing():
    full = make_clean(n=12)
    rep = validate(
        full, interval="1h",
        expected_start=full["timestamp_open"].iloc[0],
        expected_end=full["timestamp_open"].iloc[-1],
    )
    assert rep["n_missing_bars"] == 0
    assert rep["n_missing_prefix"] == 0
    assert rep["n_missing_suffix"] == 0


def test_report_includes_git_commit_key():
    rep = validate(make_clean(n=6), interval="1h")
    assert "git_commit" in rep  # str sha when in a git repo, else None


def test_validator_read_only_with_expected_range():
    df = make_clean(n=12).iloc[1:].reset_index(drop=True)
    before = df.copy()
    validate(df, interval="1h", expected_start="2020-01-01", expected_end="2020-01-01 11:00")
    pd.testing.assert_frame_equal(df, before)
