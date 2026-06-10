"""Baseline position generators (for testing the backtester only -- NOT research strategies).

Each returns a float position array p[i] in {-1,0,1}, aligned to candle bars, where p[i] is the
position decided at bar i and held over the next interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import technical_indicators as ti
from .sign_rule import _apply_mode


def _n(candles) -> int:
    return len(candles) if not isinstance(candles, int) else candles


def _finalize(pos: np.ndarray, mode: str) -> np.ndarray:
    """NaN (warm-up) -> cash, then apply long_short/long_cash/short_cash."""
    pos = np.where(np.isfinite(pos), pos, 0.0)
    return _apply_mode(pos, mode)


def buy_and_hold_long(candles) -> np.ndarray:
    return np.ones(_n(candles), dtype="float64")


def always_cash(candles) -> np.ndarray:
    return np.zeros(_n(candles), dtype="float64")


def alternating_long_cash(candles) -> np.ndarray:
    n = _n(candles)
    return np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)], dtype="float64")


def alternating_long_short(candles) -> np.ndarray:
    n = _n(candles)
    return np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)], dtype="float64")


# --- causal rule-based strategies (compute their own indicators internally) ------
def ema_crossover(candles: pd.DataFrame, *, fast: int = 12, slow: int = 48, mode: str = "long_short") -> np.ndarray:
    """+1 when fast EMA above slow EMA, -1 (or 0) below. Decided at completed bar t (causal)."""
    close = candles["close"]
    ef, es = ti.ema(close, fast), ti.ema(close, slow)
    pos = np.where(ef.to_numpy() > es.to_numpy(), 1.0, -1.0)
    pos = np.where((ef.isna() | es.isna()).to_numpy(), np.nan, pos)  # warm-up
    return _finalize(pos, mode)


def donchian_breakout(candles: pd.DataFrame, *, window: int = 24, mode: str = "long_short") -> np.ndarray:
    """+1 on close above the prior-window high, -1 below the prior-window low, else flat. Causal."""
    upper, lower = ti.donchian(candles["high"], candles["low"], window)
    close = candles["close"].to_numpy()
    pos = np.where(close > upper.to_numpy(), 1.0, np.where(close < lower.to_numpy(), -1.0, 0.0))
    pos = np.where((upper.isna() | lower.isna()).to_numpy(), np.nan, pos)
    return _finalize(pos, mode)


def rsi_mean_reversion(candles: pd.DataFrame, *, window: int = 14, low: float = 30.0,
                       high: float = 70.0, mode: str = "long_short") -> np.ndarray:
    """Mean reversion: +1 when oversold (RSI<low), -1 when overbought (RSI>high), else flat. Causal."""
    r = ti.rsi(candles["close"], window)
    rv = r.to_numpy()
    pos = np.where(rv < low, 1.0, np.where(rv > high, -1.0, 0.0))
    pos = np.where(r.isna().to_numpy(), np.nan, pos)
    return _finalize(pos, mode)
