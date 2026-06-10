"""Milestone 2.5 tests: derivatives parsers + URL builders (offline)."""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_benchmark.data.download_derivatives import (  # noqa: E402
    FUNDING_COLUMNS,
    OI_COLUMNS,
    _price_close_series,
    funding_monthly_url,
    metrics_daily_url,
    parse_funding_csv,
    parse_metrics_csv,
    price_kline_monthly_url,
)
from btc_benchmark.utils.time import epoch_to_utc  # noqa: E402


def test_url_builders():
    assert funding_monthly_url("BTCUSDT", 2024, 1) == (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip"
    )
    assert metrics_daily_url("BTCUSDT", pd.Timestamp("2024-01-05")) == (
        "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-01-05.zip"
    )
    assert price_kline_monthly_url("markPriceKlines", "BTCUSDT", "1h", 2024, 1) == (
        "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
    )


def test_parse_funding_with_header():
    csv = ("calc_time,funding_interval_hours,last_funding_rate\n"
           "1704067200000,8,0.00037409\n"
           "1704096000000,8,-0.0001\n")
    out = parse_funding_csv(io.StringIO(csv), symbol="BTCUSDT", downloaded_at="x")
    assert list(out.columns) == FUNDING_COLUMNS
    assert len(out) == 2
    assert out.iloc[0]["event_time"] == pd.Timestamp("2024-01-01", tz="UTC")
    assert out.iloc[0]["funding_rate"] == pytest.approx(0.00037409)
    assert int(out.iloc[0]["funding_interval_hours"]) == 8


def test_parse_funding_without_header():
    csv = "1704067200000,8,0.00037409\n"
    out = parse_funding_csv(io.StringIO(csv), symbol="BTCUSDT", downloaded_at="x")
    assert len(out) == 1
    assert out.iloc[0]["funding_rate"] == pytest.approx(0.00037409)


def test_parse_metrics():
    csv = ("create_time,symbol,sum_open_interest,sum_open_interest_value,"
           "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
           "count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
           "2024-01-01 00:00:00,BTCUSDT,74006.266,3131493738.8974,1.36,1.25,1.5,1.31\n"
           "2024-01-01 00:05:00,BTCUSDT,74010.0,3131500000.0,1.30,1.20,1.4,1.30\n")
    out = parse_metrics_csv(io.StringIO(csv), downloaded_at="x")
    assert set(OI_COLUMNS) <= set(out.columns)  # OI schema preserved
    # long/short positioning ratios are now also kept (large-player proxies)
    assert {"toptrader_position_ls", "toptrader_account_ls", "global_account_ls", "taker_ls_ratio"} <= set(out.columns)
    assert out.iloc[0]["event_time"] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert out.iloc[0]["open_interest"] == pytest.approx(74006.266)
    assert out.iloc[0]["open_interest_value"] == pytest.approx(3131493738.8974)
    assert out.iloc[0]["toptrader_position_ls"] == pytest.approx(1.25)
    assert out.iloc[0]["taker_ls_ratio"] == pytest.approx(1.31)


def test_price_close_series_from_zip():
    csv = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
           "taker_buy_volume,taker_buy_quote_volume,ignore\n"
           "1704067200000,42313.9,42591.9,42289.7,42503.5,0,1704070799999,0,3600,0,0,0\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("BTCUSDT-1h-2024-01.csv", csv)
    out = _price_close_series(buf.getvalue())
    assert out.iloc[0]["value"] == pytest.approx(42503.5)
    assert out.iloc[0]["event_time"] == epoch_to_utc(1704070799999)


def test_funding_event_time_dedup_sorted():
    csv = ("calc_time,funding_interval_hours,last_funding_rate\n"
           "1704096000000,8,-0.0001\n"            # out of order
           "1704067200000,8,0.0003\n"
           "1704067200000,8,0.0003\n")            # duplicate
    out = parse_funding_csv(io.StringIO(csv), symbol="BTCUSDT", downloaded_at="x")
    assert len(out) == 2  # deduped
    assert out["event_time"].is_monotonic_increasing  # sorted
