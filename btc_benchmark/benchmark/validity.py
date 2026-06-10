"""Mandatory causality gates -- run by the benchmark BEFORE scoring (failure = disqualified).

Gate 0  determinism         positions() called twice -> bit-identical (purity given fit state).
Gate 1  future-perturbation  for a set of cutoffs t0 spanning the WHOLE fold, perturb the forward
                             timeline strictly after t0 (every candle row > t0, plus any aux event /
                             sub-bar whose information lands after close[t0]); the SCORED array's
                             prefix [:t0+1] must be unchanged. The cutoffs are spread evenly across
                             [start, end) and ALWAYS include the last decision, so -- unlike the
                             audit-3 gate that stopped at 0.75n -- there is no structural tail gap;
                             the per-fold budget (`max_cutoffs`, default 512; None = exhaustive
                             stride-1) bounds the O(n)-full-evaluations cost, and the coverage
                             (exhaustive vs strided) is reported, not hidden. A 1-bar peek at index t
                             is caught by the cutoff t0 = t (it perturbs bar t+1 and still checks index
                             t); a real look-ahead bug spans many bars and trips the first cutoff it
                             reaches. Each cutoff asks for the UNIQUE window [start, t0+2) (never the
                             fixed scored window) so a strategy that memoises and replays its first
                             look-ahead output keyed by the window cannot satisfy the gate -- a fresh
                             key forces a recompute on the perturbed data. Two perturbation directions
                             (inflate + sign-flip) catch comparisons that survive inflation.
Gate 2  prefix invariance    asking for a shorter window cannot change earlier decisions (catches a
                             strategy that depends on the requested window END rather than on data).

Why index-based candle perturbation (not a timestamp compare): a 1h bar's full OHLC is known only at
its OWN close, i.e. strictly AFTER close[t0]; so EVERY candle with index > t0 is future regardless of
the exact timestamp_close convention. Perturbing by index removes a fragile dependence on a 1ms gap
between close[t] and open[t+1]. Aux frames (funding/OI/sub-bars) are NOT index-aligned to decisions,
so they are perturbed by time relative to cutoff_ts = close[t0] (a sub-bar that opens <= close[t0]
but CLOSES after it straddles the boundary and is correctly treated as future).

IMPORTANT (gate scope): the runner calls run_gates on EVERY fold's test window, so a strategy cannot
be causal on one sampled fold and cheat on the others. The holdout firewall is structural in the
runner: holdout indexes are never passed to the strategy for fit/scoring at all.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

from .contract import BenchmarkData, Strategy

# sub-bar kline frames keyed by OPEN time -> info is known only at open + interval (the bar close)
_SUBBAR_INTERVAL_MIN = {"sub1": 1, "sub5": 5, "sub15": 15}
_NUMERIC_SKIP = {"event_time", "timestamp_open", "timestamp_close"}


def _as_i8(s: pd.Series) -> np.ndarray:
    """UTC datetime series -> int64 ns since epoch (matches the backtester's bucketing convention)."""
    return pd.DatetimeIndex(pd.to_datetime(s, utc=True)).as_unit("ns").asi8


class _Perturber:
    """Reuse ONE working copy of `data` and perturb the forward region IN PLACE per cutoff, restoring
    it afterwards. Rebuilding the frames each call (esp. a ~500k-row open_interest aux) cost ~1s/call
    and made the cutoff sweep unusable; an in-place .iloc block-assignment over the perturbed suffix
    is O(suffix). Numeric columns are held as float64 so the assignment is exact.

    Perturb EVERY numeric column (candles: rows with INDEX > cutoff -- a 1h bar's full OHLC is known
    only at its OWN close, after close[t0], so index-based is robust to the timestamp_close convention;
    aux: events whose time is > close[t0]). The loader hands strategies the whole processed frame
    (quote_volume, number_of_trades, taker_buy_*, OI, premium, ...), so any of those future values is
    exploitable look-ahead and all must move. Perturbed data is fed ONLY to the (throwaway) strategy
    call, never to the scored backtest, so flipping values negative here is safe."""

    def __init__(self, data: BenchmarkData):
        c = data.candles.copy()
        self._n = len(c)
        cand_num = [col for col in c.columns if col not in _NUMERIC_SKIP and c[col].dtype.kind in "fiu"]
        for col in cand_num:
            c[col] = pd.to_numeric(c[col], errors="coerce").astype("float64")
        self._cand_pos = [c.columns.get_loc(col) for col in cand_num]
        self._cand_saved = (np.column_stack([c.iloc[:, p].to_numpy("float64") for p in self._cand_pos])
                            if self._cand_pos else None)
        self._cand = c
        self._aux_w: dict[str, pd.DataFrame] = {}
        self._aux_meta: list[tuple] = []                     # (key, ev, sorted, col_pos, saved_block)
        for k, df in data.aux.items():
            w = df.copy()
            if k in _SUBBAR_INTERVAL_MIN and "timestamp_open" in df.columns:
                ev = _as_i8(pd.to_datetime(df["timestamp_open"], utc=True)
                            + pd.Timedelta(minutes=_SUBBAR_INTERVAL_MIN[k]))  # CLOSE time of the sub-bar
            elif "event_time" in df.columns:
                ev = _as_i8(df["event_time"])
            elif "timestamp_open" in df.columns:                 # unknown frame keyed by open
                ev = _as_i8(df["timestamp_open"])
            else:
                ev = None
            num = [col for col in df.columns if col not in _NUMERIC_SKIP and df[col].dtype.kind in "fiu"]
            if ev is not None and num:
                for col in num:
                    w[col] = pd.to_numeric(w[col], errors="coerce").astype("float64")
                pos = [w.columns.get_loc(col) for col in num]
                saved = np.column_stack([w.iloc[:, p].to_numpy("float64") for p in pos])
                self._aux_meta.append((k, ev, bool(np.all(np.diff(ev) >= 0)), pos, saved))
            self._aux_w[k] = w
        self._wdata = BenchmarkData(candles=self._cand, aux=self._aux_w)

    @contextmanager
    def perturbed(self, candle_cutoff_idx: int, cutoff_i8: int, factor: float, bias: float):
        """Apply v -> v*factor + bias to every future value, yield the working data, then restore.

        Two directions per cutoff (inflate factor>0, sign-flip factor<0): any decision that is a
        monotone function of a future value takes different values under a large-positive vs a
        large-negative future, so a look-ahead cannot match the unperturbed decision under BOTH (a
        single inflate misses up-moves that stay up under inflation)."""
        regions = []                                         # (frame, row_indexer, col_pos, block) to restore
        if self._cand_saved is not None:
            sl = slice(candle_cutoff_idx + 1, self._n)
            block = self._cand_saved[sl]
            self._cand.iloc[sl, self._cand_pos] = block * factor + bias
            regions.append((self._cand, sl, self._cand_pos, block))
        for k, ev, is_sorted, pos, saved in self._aux_meta:
            w = self._aux_w[k]
            if is_sorted:
                split = int(np.searchsorted(ev, cutoff_i8, side="right"))   # events with ev > cutoff
                if split >= len(ev):
                    continue
                idx: object = slice(split, len(ev))
            else:
                idx = np.flatnonzero(ev > cutoff_i8)                        # integer positions (robust .iloc)
                if idx.size == 0:
                    continue
            block = saved[idx]
            w.iloc[idx, pos] = block * factor + bias
            regions.append((w, idx, pos, block))
        try:
            yield self._wdata
        finally:
            for frame, idx, pos, block in regions:
                frame.iloc[idx, pos] = block


# inflate + sign-flip: two extreme directions whose disagreement certifies no monotone future
# dependence (see _Perturber.perturbed). Magnitudes dominate price scale so adjacent comparisons flip.
_PERTURB_DIRECTIONS = ((3.0, 1.0), (-3.0, -1.0))


def run_gates(strategy: Strategy, data: BenchmarkData, *, start: int, end: int,
              scored: np.ndarray, max_cutoffs: int | None = 512) -> dict:
    """Validate the EXACT positions array that was scored for fold [start, end).

    CRITICAL: `scored` is the array the runner actually scored (its first positions() call for this
    fold). All gates compare against THIS array -- never an independently-recomputed one -- so a
    strategy cannot cheat on the scored call and behave honestly on the gate's calls. The runner
    calls this for EVERY fold, so passing requires causal behavior on the whole evaluated timeline.
    A window too small to run a forward/prefix check fails CLOSED (cannot be scored-but-ungated).

    `max_cutoffs` bounds the future-perturbation sweep for tractability: a black-box airtight
    (stride-1) sweep is O(n) full strategy evaluations per fold, which is minutes-to-hours on the
    real folds. With a budget the gate forward-checks `min(n, max_cutoffs)` cutoffs spread evenly
    across the WHOLE window (always reaching the last decision -- so there is no structural tail gap,
    the audit-3 failure). This catches the realistic threat (a look-ahead bug spans many bars -> hit
    at the first cutoff; tail-/segment-confined look-ahead -> hit by the even coverage) and discloses
    its granularity (`future_perturbation_exhaustive` / `_cutoffs`). Pass `max_cutoffs=None` for the
    exhaustive stride-1 certification; a sub-stride single-index adversarial peek is the only residual
    a budgeted sweep can miss, and that is reported, not hidden."""
    out: dict = {}
    base = np.asarray(scored, dtype="float64")
    n = end - start

    # determinism: re-calling positions() must reproduce the SCORED array (catches a strategy that
    # serves look-ahead on the scored call and honest positions on later calls).
    again = np.asarray(strategy.positions(data, start, end), dtype="float64")
    out["determinism"] = bool(again.shape == base.shape and np.array_equal(base, again, equal_nan=True))

    # future-perturbation: cutoffs t0 spread evenly across the WHOLE window (stride 1 when the window
    # fits the budget, else min(n, max_cutoffs) evenly-spaced points + the last index). Each cutoff
    # perturbs the entire forward timeline strictly after t0 (candles by index, aux by time) and
    # requires the scored prefix base[:t0+1] to be unchanged. The sweep always reaches the last
    # decision, so there is no structural tail gap (the audit-3 failure was cutoffs that stopped at
    # 0.75n); we early-exit on the first failure so a cheater is caught quickly.
    #
    # KEY: each cutoff asks for the UNIQUE short window [start, t0+2), NOT the fixed scored window
    # [start, end). A stateful strategy that memoises its first (scored) look-ahead output keyed by
    # (start, end) and replays it would otherwise pass every gate (the perturbed calls hit the cache);
    # a per-cutoff window key forces a real recompute on the perturbed data. Passing therefore
    # requires decision t to be invariant to BOTH future data AND the window extent past t -- exactly
    # the causal contract -- so no honest (causal, prefix-stable) strategy is ever falsely rejected.
    if n < 2:
        out["future_perturbation"] = False                       # too small to forward-check -> closed
        out["future_perturbation_exhaustive"] = False
        out["future_perturbation_cutoffs"] = 0
    else:
        if max_cutoffs is None or n <= max_cutoffs:
            cutoffs = list(range(start, end))                    # stride 1: every interior index
            exhaustive = True
        else:
            stride = -(-n // max_cutoffs)                        # ceil(n / max_cutoffs)
            cutoffs = list(range(start, end, stride))
            if cutoffs[-1] != end - 1:
                cutoffs.append(end - 1)                          # always forward-check the last decision
            exhaustive = False
        pz = _Perturber(data)
        close_i8 = _as_i8(data.candles["timestamp_close"])
        fp_ok, fp_ran = True, 0
        for t0 in cutoffs:                                       # global index of the held decision
            k = t0 - start + 1
            ew = min(t0 + 2, end)                                # unique short window -> cache cannot replay
            for factor, bias in _PERTURB_DIRECTIONS:             # inflate AND sign-flip
                with pz.perturbed(t0, int(close_i8[t0]), factor, bias) as pert:
                    p = np.asarray(strategy.positions(pert, start, ew), dtype="float64")
                fp_ran += 1
                if not (p.shape == (ew - start,) and np.array_equal(base[:k], p[:k], equal_nan=True)):
                    fp_ok = False
                    break                                        # disqualified; no need to finish
            if not fp_ok:
                break
        out["future_perturbation"] = bool(fp_ok and fp_ran > 0)
        out["future_perturbation_exhaustive"] = bool(exhaustive)
        out["future_perturbation_cutoffs"] = len(cutoffs)

    # prefix invariance: asking for a shorter window cannot change earlier (scored) decisions.
    pi_ok, pi_ran = True, 0
    for mid in (start + n // 2, end - 1):
        if mid <= start:
            continue
        pi_ran += 1
        short = np.asarray(strategy.positions(data, start, mid), dtype="float64")
        pi_ok = pi_ok and bool(short.shape == (mid - start,)
                               and np.array_equal(base[: mid - start], short, equal_nan=True))
    out["prefix_invariance"] = bool(pi_ok and pi_ran > 0)        # fail closed if no prefix ran

    out["passed"] = bool(out["determinism"] and out["future_perturbation"] and out["prefix_invariance"])
    out["gate_window"] = [int(start), int(end)]
    return out
