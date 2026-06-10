"""Sign decision rule: map a forecast (regression) or P(up) (classifier) to a position.

Stateless and causal: position at t depends only on the score at t. NaN score -> cash (0).
mode: long_short | long_cash | short_cash.
"""
from __future__ import annotations

import numpy as np

MODES = ("long_short", "long_cash", "short_cash")


def _apply_mode(pos: np.ndarray, mode: str) -> np.ndarray:
    if mode == "long_short":
        return pos
    if mode == "long_cash":
        return np.where(pos < 0, 0.0, pos)
    if mode == "short_cash":
        return np.where(pos > 0, 0.0, pos)
    raise ValueError(f"mode must be one of {MODES}")


def sign_rule_from_forecast(forecast, *, deadzone: float = 0.0, mode: str = "long_short") -> np.ndarray:
    f = np.asarray(forecast, dtype="float64")
    pos = np.where(f > deadzone, 1.0, np.where(f < -deadzone, -1.0, 0.0))
    pos = np.where(np.isfinite(f), pos, 0.0)   # NaN forecast -> cash
    return _apply_mode(pos, mode)


def sign_rule_from_proba(prob_pos, *, threshold: float = 0.0, mode: str = "long_short") -> np.ndarray:
    p = np.asarray(prob_pos, dtype="float64")
    pos = np.where(p > 0.5 + threshold, 1.0, np.where(p < 0.5 - threshold, -1.0, 0.0))
    pos = np.where(np.isfinite(p), pos, 0.0)
    return _apply_mode(pos, mode)
