"""Mandatory causality gates -- run by the benchmark BEFORE scoring (failure = disqualified).

Gate 0  determinism        positions() called twice -> bit-identical (purity given fit state)
Gate 1  future-perturbation rows > t0 (and aux events after close[t0]) perturbed -> p[:t0] unchanged
Gate 2  prefix invariance   asking for a shorter window cannot change earlier decisions
(The holdout firewall is structural in the runner: holdout indexes are never passed at all.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import BenchmarkData, Strategy

_EVENT_TIME_COLS = ("event_time", "timestamp_open")
_NUMERIC_SKIP = {"event_time", "timestamp_open", "timestamp_close"}


def _perturb_after(data: BenchmarkData, cutoff_ts: pd.Timestamp) -> BenchmarkData:
    """Return a copy with every candle row opening after `cutoff_ts` and every aux event after
    `cutoff_ts` multiplied/shifted -- past rows untouched."""
    c = data.candles.copy()
    m = pd.to_datetime(c["timestamp_open"], utc=True) > cutoff_ts
    for col in ("open", "high", "low", "close", "volume"):
        if col in c.columns:
            v = pd.to_numeric(c[col], errors="coerce").to_numpy("float64").copy()
            v[m.to_numpy()] = v[m.to_numpy()] * 1.5 + 1.0
            c[col] = v
    aux2 = {}
    for k, df in data.aux.items():
        d = df.copy()
        tcol = next((t for t in _EVENT_TIME_COLS if t in d.columns), None)
        if tcol is None:
            aux2[k] = d
            continue
        em = pd.to_datetime(d[tcol], utc=True) > cutoff_ts
        for col in d.columns:
            if col in _NUMERIC_SKIP or d[col].dtype.kind not in "fiu":
                continue
            v = pd.to_numeric(d[col], errors="coerce").to_numpy("float64").copy()
            v[em.to_numpy()] = v[em.to_numpy()] * 1.5 + 1.0
            d[col] = v
        aux2[k] = d
    return BenchmarkData(candles=c, aux=aux2)


def run_gates(strategy: Strategy, data: BenchmarkData, *, start: int, end: int) -> dict:
    """Run all gates on one (already-fit) fold window [start, end). Returns a report dict."""
    out: dict = {}
    base = np.asarray(strategy.positions(data, start, end), dtype="float64")
    again = np.asarray(strategy.positions(data, start, end), dtype="float64")
    out["determinism"] = bool(np.array_equal(base, again, equal_nan=True))

    t0 = start + (end - start) // 2
    cutoff = pd.to_datetime(data.candles["timestamp_close"].iloc[t0], utc=True)
    pert = np.asarray(strategy.positions(_perturb_after(data, cutoff), start, end), dtype="float64")
    k = t0 - start + 1
    out["future_perturbation"] = bool(np.array_equal(base[:k], pert[:k], equal_nan=True))

    # two prefix points: the half-window (catches window-statistic cheats) and end-1 (an ODD
    # offset, so parity/periodic end-dependence cannot line up with both)
    ok = True
    for mid in (start + (end - start) // 2, end - 1):
        short = np.asarray(strategy.positions(data, start, mid), dtype="float64")
        ok = ok and bool(np.array_equal(base[: mid - start], short, equal_nan=True))
    out["prefix_invariance"] = ok

    out["passed"] = bool(out["determinism"] and out["future_perturbation"] and out["prefix_invariance"])
    out["gate_window"] = [int(start), int(end)]
    return out
