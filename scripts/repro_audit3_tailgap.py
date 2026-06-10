"""Audit-3 regression repro on the REAL bundle with the REAL default SPLIT.

Before the fix, a look-ahead cheater confined to the tail (last 25%) of each fold scored
net +63,151.9% / Sharpe 4.927 with disqualified=False, because future_perturbation only
forward-checked cutoffs at 0.25/0.5/0.75n and never the tail. After the stride-1 fix, the
same cheater must be DISQUALIFIED (future_perturbation=False) on the real data + real split.

Run:  python -m scripts.repro_audit3_tailgap
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btc_benchmark.benchmark import load_benchmark_data, run_benchmark


class TailGapCheater:
    """Honest causal on the first 75% of every fold; peeks one bar ahead in the tail."""
    name, horizon = "tail_gap_cheater", 1

    def fit(self, data, train_start, train_end):
        pass

    def positions(self, data, start, end):
        c = pd.to_numeric(data.candles["close"], errors="coerce").to_numpy("float64")
        n = end - start
        cut = start + int(n * 0.75)
        ema = pd.Series(c[:end]).ewm(span=24, adjust=False).mean().to_numpy()
        nxt = np.concatenate([c[1:], [c[-1]]])
        out = np.empty(n)
        for k, t in enumerate(range(start, end)):
            out[k] = (1.0 if c[t] > ema[t] else 0.0) if t <= cut \
                else (1.0 if nxt[t] > c[t] else 0.0)
        return out


def main() -> None:
    data = load_benchmark_data(".", include_sub_bars=False)
    rep = run_benchmark(TailGapCheater(), data, team="audit3_repro")
    g = rep["gates"]
    print("=== audit-3 tail-gap cheater on REAL data + REAL default SPLIT ===")
    print(f"net               : {rep['net']*100:,.1f}%")
    print(f"sharpe            : {rep['sharpe']}")
    print(f"future_perturbation: {g['future_perturbation']}")
    print(f"failed_folds      : {g['failed_folds']}")
    print(f"disqualified      : {rep['disqualified']}")
    assert rep["disqualified"] is True, "FIX REGRESSED: tail-gap cheater was NOT disqualified"
    assert g["future_perturbation"] is False
    print("\nOK: tail-gap look-ahead is now disqualified (was net +63,151.9%, dq=False pre-fix).")


if __name__ == "__main__":
    main()
