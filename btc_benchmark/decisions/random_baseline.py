"""Random turnover-matched baseline.

Generates random position paths from {-1,0,1} with approximately the same number of position
changes as a reference strategy, across many fixed-seed trials. Used to test whether a strategy
beats a random strategy of similar trading frequency (not just one that trades less).
"""
from __future__ import annotations

import numpy as np


def count_changes(positions) -> int:
    p = np.asarray(positions, dtype="float64")
    if len(p) == 0:
        return 0
    prev = np.concatenate([[0.0], p[:-1]])
    return int(np.sum(p != prev))


def random_turnover_matched_positions(
    n: int, n_changes: int, *, allowed=(-1.0, 0.0, 1.0), n_trials: int = 100, seed: int = 0
) -> list[np.ndarray]:
    """Return `n_trials` position arrays, each with ~`n_changes` position changes. Deterministic."""
    n_changes = max(0, min(int(n_changes), n))
    trials: list[np.ndarray] = []
    for k in range(n_trials):
        rng = np.random.default_rng(seed + k)
        pos = np.zeros(n, dtype="float64")
        change_points = set(rng.choice(n, size=n_changes, replace=False).tolist()) if n_changes else set()
        cur = 0.0
        for t in range(n):
            if t in change_points:
                choices = [x for x in allowed if x != cur]
                cur = float(rng.choice(choices))
            pos[t] = cur
        trials.append(pos)
    return trials


def aggregate_trial_metrics(metric_dicts: list[dict], keys: list[str]) -> dict:
    """mean / median / p5 / p95 across trials for the given metric keys."""
    out: dict = {}
    for key in keys:
        vals = np.array([m.get(key, np.nan) for m in metric_dicts], dtype="float64")
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[key] = {"mean": None, "median": None, "p5": None, "p95": None}
        else:
            out[key] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p5": float(np.percentile(vals, 5)),
                "p95": float(np.percentile(vals, 95)),
            }
    return out
