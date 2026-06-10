"""Causal technical indicators.

EVERY indicator here is strictly causal: the value at bar t uses only information from
bars <= t (trailing rolling windows, recursive EWMs, and backward shifts). There are NO
centered windows and NO forward shifts. This is enforced by the generic causality test in
tests/test_indicators_causality.py (perturbing bars after t must not change features at t).

All functions take/return pandas objects aligned to the input index; warm-up periods are
NaN (via min_periods) rather than back-filled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RETURN_WINDOWS = (1, 3, 6, 12, 24, 48, 72)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """a / b with zero denominators -> NaN (honest, not 0)."""
    b = b.where(b != 0, np.nan)
    return a / b


# --- returns / volatility -------------------------------------------------------
def simple_return(close: pd.Series, w: int = 1) -> pd.Series:
    return close / close.shift(w) - 1.0


def log_return(close: pd.Series, w: int = 1) -> pd.Series:
    prev = close.shift(w)
    return np.log(_safe_div(close, prev))


def rolling_volatility(close: pd.Series, w: int) -> pd.Series:
    """Std of 1-step log returns over a trailing window (realized volatility proxy)."""
    r = log_return(close, 1)
    return r.rolling(w, min_periods=w).std()


# --- moving averages ------------------------------------------------------------
def sma(close: pd.Series, w: int) -> pd.Series:
    return close.rolling(w, min_periods=w).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def ema_distance(close: pd.Series, span: int) -> pd.Series:
    e = ema(close, span)
    return _safe_div(close, e) - 1.0


def ema_slope(close: pd.Series, span: int, k: int = 1) -> pd.Series:
    e = ema(close, span)
    return _safe_div(e - e.shift(k), e.shift(k))


# --- momentum -------------------------------------------------------------------
def rsi(close: pd.Series, w: int = 14) -> pd.Series:
    """Wilder RSI (recursive EWM, causal)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean()
    avg_loss = loss.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean()
    rs = _safe_div(avg_gain, avg_loss)
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 (only gains) -> RSI 100; avg_gain == 0 (only losses) -> RSI 0
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)  # flat -> neutral
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram)."""
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


# --- volatility / range ---------------------------------------------------------
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, w: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean()


def atr_ratio(high: pd.Series, low: pd.Series, close: pd.Series, w: int = 14) -> pd.Series:
    return _safe_div(atr(high, low, close, w), close)


def bollinger(close: pd.Series, w: int = 20, k: float = 2.0):
    """Return (mid, upper, lower, std)."""
    mid = close.rolling(w, min_periods=w).mean()
    std = close.rolling(w, min_periods=w).std()
    return mid, mid + k * std, mid - k * std, std


def bollinger_width(close: pd.Series, w: int = 20, k: float = 2.0) -> pd.Series:
    mid, upper, lower, _ = bollinger(close, w, k)
    return _safe_div(upper - lower, mid)


def bollinger_zscore(close: pd.Series, w: int = 20) -> pd.Series:
    mid = close.rolling(w, min_periods=w).mean()
    std = close.rolling(w, min_periods=w).std()
    return _safe_div(close - mid, std)


# --- channels / breakout --------------------------------------------------------
def donchian(high: pd.Series, low: pd.Series, w: int):
    """Prior-window Donchian channel (shifted by 1 so breakout is meaningful). Causal."""
    upper = high.shift(1).rolling(w, min_periods=w).max()
    lower = low.shift(1).rolling(w, min_periods=w).min()
    return upper, lower


def donchian_breakout(high: pd.Series, low: pd.Series, close: pd.Series, w: int) -> pd.Series:
    """Distance above prior-window upper (positive) or below lower (negative); else 0."""
    upper, lower = donchian(high, low, w)
    above = _safe_div(close - upper, upper)
    below = _safe_div(close - lower, lower)
    out = pd.Series(0.0, index=close.index)
    out = out.where(~(close > upper), above)
    out = out.where(~(close < lower), below)
    # keep NaN during warm-up
    return out.where(upper.notna() & lower.notna(), np.nan)


def rolling_high_distance(high: pd.Series, close: pd.Series, w: int) -> pd.Series:
    return _safe_div(close, high.rolling(w, min_periods=w).max()) - 1.0


def rolling_low_distance(low: pd.Series, close: pd.Series, w: int) -> pd.Series:
    return _safe_div(close, low.rolling(w, min_periods=w).min()) - 1.0


def rolling_max_drawdown(close: pd.Series, w: int) -> pd.Series:
    """Drawdown of close from its trailing-w-window high (<= 0)."""
    roll_max = close.rolling(w, min_periods=w).max()
    return _safe_div(close, roll_max) - 1.0


# --- volume ---------------------------------------------------------------------
def volume_zscore(volume: pd.Series, w: int) -> pd.Series:
    mean = volume.rolling(w, min_periods=w).mean()
    std = volume.rolling(w, min_periods=w).std()
    return _safe_div(volume - mean, std)


def relative_volume(volume: pd.Series, w: int) -> pd.Series:
    return _safe_div(volume, volume.rolling(w, min_periods=w).mean())


# --- candle shape (per-bar, trivially causal) -----------------------------------
def candle_body_ratio(o: pd.Series, h: pd.Series, low: pd.Series, c: pd.Series) -> pd.Series:
    rng = h - low
    return _safe_div(c - o, rng)


def upper_wick_ratio(o: pd.Series, h: pd.Series, low: pd.Series, c: pd.Series) -> pd.Series:
    rng = h - low
    return _safe_div(h - np.maximum(o, c), rng)


def lower_wick_ratio(o: pd.Series, h: pd.Series, low: pd.Series, c: pd.Series) -> pd.Series:
    rng = h - low
    return _safe_div(np.minimum(o, c) - low, rng)
