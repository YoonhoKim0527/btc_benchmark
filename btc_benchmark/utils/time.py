"""UTC time and epoch-timestamp handling.

Binance gotcha this module exists to handle: spot kline timestamps switched from
milliseconds to MICROSECONDS on 2025-01-01, while USD-M futures stayed in milliseconds.
We never assume a unit — we detect it from magnitude, per series/file.
"""
from __future__ import annotations

import pandas as pd

# canonical interval -> milliseconds
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def detect_epoch_unit(value: int) -> str:
    """Detect the epoch unit of an integer timestamp by magnitude.

    For dates in 2017-2026: seconds ~1.5e9, ms ~1.5e12, us ~1.5e15, ns ~1.5e18.
    """
    v = abs(int(value))
    if v >= 1e17:
        return "ns"
    if v >= 1e14:
        return "us"
    if v >= 1e11:
        return "ms"
    if v >= 1e8:
        return "s"
    raise ValueError(f"cannot detect epoch unit for value={value!r}")


def epoch_to_utc(value: int, unit: str | None = None) -> pd.Timestamp:
    """Convert a single epoch integer to a tz-aware UTC Timestamp."""
    if unit is None:
        unit = detect_epoch_unit(value)
    return pd.Timestamp(int(value), unit=unit, tz="UTC")


def epoch_series_to_utc(s: "pd.Series") -> "pd.Series":
    """Convert a Series of epoch integers to UTC datetimes, detecting the unit PER ELEMENT.

    Per-element detection (by magnitude) is required because concatenating Binance
    monthly files can mix units within one Series: spot timestamps switch from ms to
    MICROSECONDS on 2025-01-01 while futures stay ms. Detecting the unit once for the
    whole series would silently misplace the minority rows (e.g. ms rows interpreted as
    us land in 1970, or us rows as ms overflow) — exactly the silent corruption the
    project forbids. Conversion is done per unit-group via pandas' exact int64 path
    (no float precision loss for s/ms/us magnitudes). Unrecognizable values -> NaT.
    """
    s = pd.Series(s)
    iv = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    av = iv.abs()
    bands = [("s", 1e8, 1e11), ("ms", 1e11, 1e14), ("us", 1e14, 1e17), ("ns", 1e17, float("inf"))]
    for unit, lo, hi in bands:
        mask = (av >= lo) & (av < hi)
        if bool(mask.any()):
            out.loc[mask] = pd.to_datetime(iv[mask].astype("int64"), unit=unit, utc=True)
    return out


def interval_timedelta(interval: str) -> pd.Timedelta:
    if interval not in INTERVAL_MS:
        raise KeyError(f"unknown interval {interval!r}; known: {sorted(INTERVAL_MS)}")
    return pd.Timedelta(milliseconds=INTERVAL_MS[interval])


def build_regular_grid(start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DatetimeIndex:
    """Inclusive regular UTC grid of bar-open timestamps at the given interval."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tz is None:
        start = start.tz_localize("UTC")
    if end.tz is None:
        end = end.tz_localize("UTC")
    return pd.date_range(start=start, end=end, freq=interval_timedelta(interval), tz="UTC")


def close_time_from_open(open_ts: "pd.Series", interval: str) -> "pd.Series":
    """Bar-close timestamp = open + interval - 1ms (Binance's inclusive convention)."""
    return pd.to_datetime(open_ts, utc=True) + interval_timedelta(interval) - pd.Timedelta(milliseconds=1)
