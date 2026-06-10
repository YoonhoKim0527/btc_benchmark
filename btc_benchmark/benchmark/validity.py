"""Mandatory causality gates -- run by the benchmark BEFORE scoring (failure = disqualified).

Gate 0  determinism        positions() called twice -> bit-identical (purity given fit state)
Gate 1  future-perturbation for several cutoffs t0, perturb EVERY bar/aux event whose information
                            is not yet known at close[t0]; positions[:t0+1] must be unchanged. Sub-bar
                            frames are perturbed by their CLOSE time (open + interval), so a sub-bar
                            that opens <= close[t0] but closes after it is correctly treated as future.
Gate 2  prefix invariance   asking for a shorter window cannot change earlier decisions

IMPORTANT (gate scope): the runner calls run_gates on EVERY fold's test window, so a strategy cannot
be causal on one sampled fold and cheat on the others. (The holdout firewall is structural in the
runner: holdout indexes are never passed to the strategy at all.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import BenchmarkData, Strategy

# aux frames keyed by their EVENT time (the instant the info becomes known)
_EVENT_TIME_AUX = {"funding", "open_interest", "mark_premium", "premium"}
# sub-bar kline frames keyed by OPEN time -> info is known only at open + interval (the bar close)
_SUBBAR_INTERVAL_MIN = {"sub1": 1, "sub5": 5, "sub15": 15}
_NUMERIC_SKIP = {"event_time", "timestamp_open", "timestamp_close"}


def _perturb_after(data: BenchmarkData, cutoff_ts: pd.Timestamp) -> BenchmarkData:
    """Copy with every value NOT known by close[t0]=cutoff_ts perturbed; values known by then untouched.

    - candles: a 1h bar opening after cutoff is future (its own close > cutoff).
    - event-time aux (funding/OI/premium): event_time > cutoff is future.
    - sub-bar klines (1m/5m/15m): a bar whose CLOSE (open + interval) > cutoff is future, even if it
      OPENED at/<=cutoff (it straddles the boundary and encodes post-cutoff price)."""
    c = data.candles.copy()
    m = pd.to_datetime(c["timestamp_open"], utc=True) > cutoff_ts
    for col in ("open", "high", "low", "close", "volume"):
        if col in c.columns:
            v = pd.to_numeric(c[col], errors="coerce").to_numpy("float64").copy()
            v[m.to_numpy()] = v[m.to_numpy()] * 1.5 + 1.0
            c[col] = v
    aux2: dict[str, pd.DataFrame] = {}
    for k, df in data.aux.items():
        d = df.copy()
        if k in _SUBBAR_INTERVAL_MIN and "timestamp_open" in d.columns:
            close_t = pd.to_datetime(d["timestamp_open"], utc=True) + pd.Timedelta(
                minutes=_SUBBAR_INTERVAL_MIN[k])
            em = close_t > cutoff_ts
        elif "event_time" in d.columns:
            em = pd.to_datetime(d["event_time"], utc=True) > cutoff_ts
        elif "timestamp_open" in d.columns:        # unknown frame keyed by open -> treat open as event
            em = pd.to_datetime(d["timestamp_open"], utc=True) > cutoff_ts
        else:
            aux2[k] = d
            continue
        emn = em.to_numpy()
        for col in d.columns:
            if col in _NUMERIC_SKIP or d[col].dtype.kind not in "fiu":
                continue
            v = pd.to_numeric(d[col], errors="coerce").to_numpy("float64").copy()
            v[emn] = v[emn] * 1.5 + 1.0
            d[col] = v
        aux2[k] = d
    return BenchmarkData(candles=c, aux=aux2)


def run_gates(strategy: Strategy, data: BenchmarkData, *, start: int, end: int) -> dict:
    """Run all gates on one (already-fit) fold window [start, end). Returns a report dict.

    The runner calls this for EVERY fold (see runner.run_benchmark), so passing requires causal
    behavior on the whole evaluated timeline, not one sampled window."""
    out: dict = {}
    base = np.asarray(strategy.positions(data, start, end), dtype="float64")
    again = np.asarray(strategy.positions(data, start, end), dtype="float64")
    out["determinism"] = bool(np.array_equal(base, again, equal_nan=True))

    # future-perturbation at several interior cutoffs (not just the midpoint), each perturbing the
    # WHOLE forward timeline + straddling sub-bars
    fp_ok = True
    n = end - start
    for frac in (0.25, 0.5, 0.75):
        t0 = start + int(n * frac)
        if not (start < t0 < end):
            continue
        cutoff = pd.to_datetime(data.candles["timestamp_close"].iloc[t0], utc=True)
        pert = np.asarray(strategy.positions(_perturb_after(data, cutoff), start, end), "float64")
        k = t0 - start + 1
        fp_ok = fp_ok and bool(np.array_equal(base[:k], pert[:k], equal_nan=True))
    out["future_perturbation"] = fp_ok

    # prefix invariance at two offsets (half-window catches window-statistic cheats; end-1 odd offset
    # so parity/periodic end-dependence cannot line up with both)
    pi_ok = True
    for mid in (start + n // 2, end - 1):
        if mid <= start:
            continue
        short = np.asarray(strategy.positions(data, start, mid), dtype="float64")
        pi_ok = pi_ok and bool(np.array_equal(base[: mid - start], short, equal_nan=True))
    out["prefix_invariance"] = pi_ok

    out["passed"] = bool(out["determinism"] and out["future_perturbation"] and out["prefix_invariance"])
    out["gate_window"] = [int(start), int(end)]
    return out
