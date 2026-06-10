"""Milestone 2 tests: 1m -> higher-timeframe resampling (offline, synthetic)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_benchmark.data.resample import compare_direct_vs_resampled, resample  # noqa: E402
from btc_benchmark.data.schema import DataQualityFlag, PROCESSED_COLUMNS  # noqa: E402

DLA = "2026-01-01T00:00:00+00:00"


def make_1m(minutes, start="2020-01-01"):
    """1m candles at the given integer minute offsets (allows gaps)."""
    base = pd.Timestamp(start, tz="UTC")
    ts = pd.DatetimeIndex([base + pd.Timedelta(minutes=m) for m in minutes])
    close = 100.0 + np.array(minutes, dtype=float)  # strictly increasing, gap-aware
    return pd.DataFrame({
        "timestamp_open": ts,
        "open": close - 0.1,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.ones(len(minutes)),
        "quote_volume": np.full(len(minutes), 10.0),
        "number_of_trades": np.full(len(minutes), 2),
        "taker_buy_base_volume": np.full(len(minutes), 0.5),
        "taker_buy_quote_volume": np.full(len(minutes), 5.0),
    })


def _r(df, target, **kw):
    return resample(df, target_interval=target, symbol="BTCUSDT", market_type="futures_um",
                    downloaded_at=DLA, **kw)


def test_5m_aggregation_correct():
    df = make_1m(range(10))  # 00:00..00:09 -> two complete 5m bars
    out, rep = _r(df, "5m")
    assert list(out.columns) == PROCESSED_COLUMNS
    assert len(out) == 2
    assert rep["expected_subbars_per_bar"] == 5
    b0 = out.iloc[0]
    assert b0["open"] == 99.9                  # first open (minute 0)
    assert b0["high"] == 100.0 + 4 + 1.0       # max over minutes 0..4
    assert b0["low"] == 100.0 - 1.0            # min over minutes 0..4
    assert b0["close"] == 104.0                # last close (minute 4)
    assert b0["volume"] == 5.0
    assert int(b0["number_of_trades"]) == 10
    assert b0["data_quality_flag"] == DataQualityFlag.NORMAL
    assert int(b0["missing_subbar_count"]) == 0
    assert out.iloc[1]["close"] == 109.0       # minute 9


def test_1h_aggregation_correct():
    df = make_1m(range(60))
    out, rep = _r(df, "1h")
    assert len(out) == 1
    assert out.iloc[0]["open"] == 99.9
    assert out.iloc[0]["close"] == 159.0
    assert out.iloc[0]["volume"] == 60.0
    assert int(out.iloc[0]["number_of_trades"]) == 120


def test_interval_boundary_no_off_by_one():
    df = make_1m(range(6))  # minutes 0..5 -> bar0 complete (0-4), bar1 partial (just minute 5)
    out, _ = _r(df, "5m")
    assert len(out) == 2
    assert str(out.iloc[0]["timestamp_open"]) == "2020-01-01 00:00:00+00:00"
    assert str(out.iloc[1]["timestamp_open"]) == "2020-01-01 00:05:00+00:00"
    assert out.iloc[0]["data_quality_flag"] == DataQualityFlag.NORMAL  # minute 4 in bar0, not bar1


def test_missing_subbar_marks_partial_interval():
    df = make_1m([0, 1, 3, 4, 5, 6, 7, 8, 9])  # minute 2 missing -> first 5m bar has 4 sub-bars
    out, rep = _r(df, "5m")
    b0 = out.iloc[0]
    assert b0["data_quality_flag"] == DataQualityFlag.PARTIAL_INTERVAL
    assert int(b0["missing_subbar_count"]) == 1
    assert b0["volume"] == 4.0
    assert rep["partial_intervals"] == 1


def test_fully_missing_interval_flat_filled_when_policy_allows():
    df = make_1m(list(range(5)) + list(range(10, 15)))  # 00:05..00:09 fully missing
    out, rep = _r(df, "5m", missing_bar_policy="flat_bar_fill")
    assert len(out) == 3
    mid = out.iloc[1]
    assert bool(mid["is_imputed"]) is True
    assert mid["data_quality_flag"] == DataQualityFlag.MISSING_FILLED
    assert mid["open"] == mid["high"] == mid["low"] == mid["close"] == out.iloc[0]["close"]
    assert mid["volume"] == 0.0
    assert int(mid["missing_subbar_count"]) == 5
    assert rep["full_missing_intervals"] == 1


def test_fully_missing_interval_dropped_under_strict_drop():
    df = make_1m(list(range(5)) + list(range(10, 15)))
    out, rep = _r(df, "5m", missing_bar_policy="strict_drop")
    assert len(out) == 2  # the empty 00:05 bar is dropped
    assert not out["is_imputed"].any()
    assert rep["full_missing_intervals"] == 0


def test_imputed_subbars_excluded_by_default():
    df = make_1m(range(10))
    df["is_imputed"] = [False] * 10
    df.loc[2, "is_imputed"] = True  # minute 2 imputed -> excluded -> bar0 partial
    out, rep = _r(df, "5m")  # allow_resample_from_imputed=False default
    assert rep["n_excluded_imputed_subbars"] == 1
    assert out.iloc[0]["data_quality_flag"] == DataQualityFlag.PARTIAL_INTERVAL
    # with allow=True the imputed sub-bar is included -> complete
    out2, rep2 = _r(df, "5m", allow_resample_from_imputed=True)
    assert rep2["n_excluded_imputed_subbars"] == 0
    assert out2.iloc[0]["data_quality_flag"] == DataQualityFlag.NORMAL


def test_compare_direct_vs_resampled():
    df = make_1m(range(20))
    res, _ = _r(df, "5m")
    direct = res.copy()  # identical -> no mismatch
    rep = compare_direct_vs_resampled(direct, res)
    assert rep["n_common"] == len(res)
    assert all(f["n_mismatch"] == 0 for f in rep["fields"].values())
    # perturb one close -> mismatch detected
    direct2 = res.copy()
    direct2.loc[0, "close"] = direct2.loc[0, "close"] * 1.5
    rep2 = compare_direct_vs_resampled(direct2, res)
    assert rep2["fields"]["close"]["n_mismatch"] == 1


def test_resample_does_not_mutate_input():
    df = make_1m(range(10))
    before = df.copy()
    _r(df, "5m")
    pd.testing.assert_frame_equal(df, before)


# --- hardening tests added after code review ----------------------------------
def test_duplicate_subbars_deduped_not_double_counted():
    df = make_1m(range(5))                       # one complete 5m bar
    dup = pd.concat([df, df], ignore_index=True)  # every 1m timestamp duplicated
    out, rep = _r(dup, "5m")
    assert rep["n_duplicate_subbars"] == 5
    assert len(out) == 1
    assert out.iloc[0]["volume"] == 5.0                    # NOT 10 (no double count)
    assert int(out.iloc[0]["number_of_trades"]) == 10      # 5 bars x 2 trades
    assert int(out.iloc[0]["missing_subbar_count"]) == 0   # never negative
    assert out.iloc[0]["data_quality_flag"] == DataQualityFlag.NORMAL


def test_offgrid_subbars_dropped_and_counted():
    df = make_1m(range(10))
    extra = df.iloc[[2]].copy()
    extra["timestamp_open"] = extra["timestamp_open"] + pd.Timedelta(seconds=30)  # off the 1m grid
    df2 = pd.concat([df, extra], ignore_index=True)
    out, rep = _r(df2, "5m")
    assert rep["n_offgrid_subbars"] == 1
    assert out.iloc[0]["data_quality_flag"] == DataQualityFlag.NORMAL  # 5 real on-grid sub-bars


def test_all_nan_volume_bucket_is_nan_not_silent_zero():
    df = make_1m(range(5))
    df["volume"] = np.nan  # processed-frame edge: missing volume
    out, _ = _r(df, "5m")
    assert pd.isna(out.iloc[0]["volume"])  # NaN, not a fabricated 0
