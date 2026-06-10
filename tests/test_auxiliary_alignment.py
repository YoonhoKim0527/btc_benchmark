"""Milestone 2.5 tests: backward as-of alignment (causality, boundary, staleness)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_benchmark.data.align_auxiliary import (  # noqa: E402
    asof_attach,
    attach_funding,
    attach_open_interest,
    funding_events_crossing,
)


def candles(close_times):
    ts = pd.to_datetime(close_times, utc=True, format="mixed")
    return pd.DataFrame({"timestamp_close": ts, "close": np.arange(len(ts), dtype=float) + 100})


def events(times, **cols):
    df = pd.DataFrame({"event_time": pd.to_datetime(times, utc=True, format="mixed")})
    for k, v in cols.items():
        df[k] = v
    return df


def test_exact_boundary_and_no_future():
    c = candles(["2024-01-01 01:00:00", "2024-01-01 02:00:00"])
    # event0 EXACTLY at candle0's close; event1 1ms AFTER candle1's close (future)
    e = events(["2024-01-01 01:00:00", "2024-01-01 02:00:00.001"], funding_rate=[0.001, 0.002])
    out = asof_attach(c, e, value_cols=["funding_rate"], age_col="funding_age_seconds")
    # candle0: event_time == close -> attached
    assert out.loc[0, "funding_rate"] == pytest.approx(0.001)
    assert out.loc[0, "funding_age_seconds"] == 0.0
    # candle1: event1 is in the FUTURE (close+1ms) -> NOT attached; falls back to event0
    assert out.loc[1, "funding_rate"] == pytest.approx(0.001)
    assert out.loc[1, "funding_age_seconds"] == pytest.approx(3600.0)


def test_no_event_before_candle_is_nan():
    c = candles(["2024-01-01 01:00:00", "2024-01-01 02:00:00"])
    e = events(["2024-01-01 05:00:00"], funding_rate=[0.001])  # only a future event
    out = asof_attach(c, e, value_cols=["funding_rate"], age_col="funding_age_seconds")
    assert pd.isna(out.loc[0, "funding_rate"])
    assert pd.isna(out.loc[0, "funding_age_seconds"])


def test_staleness_flag_and_nan():
    c = candles(["2024-01-01 10:00:00"])
    e = events(["2024-01-01 00:00:00"], funding_rate=[0.001])  # 10h stale
    out = attach_funding(c, e, staleness_threshold_seconds=3600)  # 1h threshold
    assert pd.isna(out.loc[0, "funding_rate"])      # stale -> NaN
    assert bool(out.loc[0, "funding_stale"]) is True
    assert out.loc[0, "funding_age_seconds"] == pytest.approx(36000.0)


def test_missing_aux_does_not_break_pipeline():
    c = candles(["2024-01-01 01:00:00", "2024-01-01 02:00:00"])
    empty = pd.DataFrame(columns=["event_time", "funding_rate"])
    out = asof_attach(c, empty, value_cols=["funding_rate"], age_col="funding_age_seconds")
    assert len(out) == 2
    assert out["funding_rate"].isna().all()


def test_open_interest_attach_columns():
    c = candles(["2024-01-01 01:00:00"])
    e = events(["2024-01-01 00:30:00"], open_interest=[1000.0], open_interest_value=[5e7])
    out = attach_open_interest(c, e)
    assert out.loc[0, "open_interest"] == 1000.0
    assert out.loc[0, "open_interest_age_seconds"] == pytest.approx(1800.0)


def test_funding_events_crossing_is_event_based_not_per_candle():
    f = events(["2024-01-01 00:00:00", "2024-01-01 08:00:00", "2024-01-01 16:00:00"],
               funding_rate=[0.0001, 0.0002, 0.0003])
    # a position held from 00:00 to 08:00 crosses ONLY the 08:00 event (not 00:00, which is t0)
    crossed = funding_events_crossing(f, "2024-01-01 00:00:00", "2024-01-01 08:00:00")
    assert len(crossed) == 1
    assert crossed.loc[0, "event_time"] == pd.Timestamp("2024-01-01 08:00:00", tz="UTC")
    # held across nothing -> empty (no per-candle funding)
    assert len(funding_events_crossing(f, "2024-01-01 00:00:00", "2024-01-01 07:59:59")) == 0


def test_asof_does_not_mutate_inputs():
    c = candles(["2024-01-01 01:00:00", "2024-01-01 02:00:00"])
    e = events(["2024-01-01 00:30:00"], funding_rate=[0.001])
    cb, eb = c.copy(), e.copy()
    asof_attach(c, e, value_cols=["funding_rate"])
    pd.testing.assert_frame_equal(c, cb)
    pd.testing.assert_frame_equal(e, eb)
