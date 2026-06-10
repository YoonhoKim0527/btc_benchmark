"""Combinatorial Purged Cross-Validation (López de Prado, Advances in Financial ML ch.7 / 12).

Standard walk-forward yields exactly ONE out-of-sample path -> one Sharpe, no distribution, so you
cannot tell a robust edge from a single lucky ordering. CPCV partitions the development timeline into
N contiguous groups, then for EVERY combination of `k_test` groups, trains on the remaining N-k groups
(PURGED of label-overlap and EMBARGOED around each test group) and predicts the k held-out groups.
Each group is thereby predicted C(N-1, k-1) times by models trained on different data; assembling one
prediction per group in time order reconstructs ``phi = C(N-1, k-1)`` full-length OOS PATHS -> a
DISTRIBUTION of OOS performance (mean / std / percentiles, fraction of losing paths) instead of a
point estimate.

Scope & honesty:
  - This re-evaluates a SINGLE, already-chosen configuration more honestly. It does NOT select among
    configurations, so it adds no multiple-testing / selection bias of its own (that is what the
    Deflated Sharpe and PBO-CSCV are for).
  - Strictly causal: future bars are used only as training labels. Purge + embargo remove label
    leakage and the serial-correlation halo around each test group on BOTH sides -- interior test
    groups straddle the training data, so this is the combinatorial generalization of the
    one-directional walk-forward purge in ``walk_forward.py``.
  - Group-only: this module is pure index arithmetic (no model, no pandas). Callers supply a
    fit/predict callback and the assembled paths are scored with the existing backtester, so per-path
    accounting is IDENTICAL to the walk-forward (no accounting code duplicated). The CPCV and
    walk-forward Sharpes are NOT like-for-like point estimates, though: CPCV paths cover the FULL
    development region while the walk-forward covers only its later test windows -- read it as
    distribution-vs-point, not as a direct head-to-head.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np


@dataclass
class CPCVSplit:
    """One combinatorial fold: which groups are test, the (post-purge) train indices, test slices."""
    combo_id: int
    test_groups: tuple[int, ...]
    train_idx: np.ndarray                 # sorted training sample indices, after purge + embargo
    test_slices: list[tuple[int, int]]    # [start, stop) per test group, in time order


def make_groups(n_samples: int, n_groups: int) -> list[tuple[int, int]]:
    """Partition ``[0, n_samples)`` into ``n_groups`` contiguous, near-equal ``[start, stop)`` slices."""
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if n_groups > n_samples:
        raise ValueError(f"n_groups ({n_groups}) > n_samples ({n_samples})")
    edges = np.linspace(0, n_samples, n_groups + 1).round().astype(int)
    groups = [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]
    if any(stop <= start for start, stop in groups):
        raise ValueError("degenerate (empty) group -- reduce n_groups")
    return groups


def n_paths(n_groups: int, k_test: int) -> int:
    """Number of reconstructable backtest paths phi = C(N-1, k-1)."""
    return comb(n_groups - 1, k_test - 1)


def cpcv_splits(n_samples: int, *, n_groups: int, k_test: int, horizon: int,
                embargo: int | None = None) -> list[CPCVSplit]:
    """All C(N, k) combinatorial folds with purge (label horizon) + embargo around each test group.

    For a fixed-horizon label, train sample ``i`` carries label endpoint ``i + horizon``; it leaks if
    ``[i, i+horizon]`` intersects any test span ``[ts, te)`` -> purge ``i in [ts-horizon, te)``. The
    embargo additionally drops ``[te, te+embargo)`` (serial-correlation halo just after each test
    group). ``embargo`` defaults to ``horizon``. Test samples are never in train by construction.
    """
    if k_test < 1 or k_test >= n_groups:
        raise ValueError("require 1 <= k_test < n_groups")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    emb = horizon if embargo is None else int(embargo)
    if emb < 0:
        raise ValueError("embargo must be >= 0")
    groups = make_groups(n_samples, n_groups)
    splits: list[CPCVSplit] = []
    for combo_id, combo in enumerate(combinations(range(n_groups), k_test)):
        test_slices = [groups[g] for g in combo]              # combo is sorted -> slices in time order
        test_mask = np.zeros(n_samples, dtype=bool)
        purge_mask = np.zeros(n_samples, dtype=bool)
        for ts, te in test_slices:
            test_mask[ts:te] = True
            purge_mask[max(0, ts - horizon):te] = True        # left: train labels reaching into test
            purge_mask[te:min(n_samples, te + emb)] = True    # right: embargo
        train_idx = np.where(~test_mask & ~purge_mask)[0]
        splits.append(CPCVSplit(combo_id, tuple(combo), train_idx, test_slices))
    return splits


def assemble_paths(splits: list[CPCVSplit], *, n_groups: int, k_test: int) -> list[list[tuple[int, int]]]:
    """Reconstruct phi backtest paths. Each path is ``[(group_id, combo_id), ...]`` over ALL N groups
    in time order; path ``p`` assigns group ``g`` the prediction from the ``p``-th combo in which ``g``
    was a test member. The phi paths partition every (combo, test-group) prediction exactly once."""
    phi = n_paths(n_groups, k_test)
    per_group: dict[int, list[int]] = {g: [] for g in range(n_groups)}
    for s in splits:
        for g in s.test_groups:
            per_group[g].append(s.combo_id)
    for g in range(n_groups):
        if len(per_group[g]) != phi:                          # invariant: each group tested phi times
            raise ValueError(f"group {g} appears {len(per_group[g])} times, expected phi={phi}")
    return [[(g, per_group[g][p]) for g in range(n_groups)] for p in range(phi)]
