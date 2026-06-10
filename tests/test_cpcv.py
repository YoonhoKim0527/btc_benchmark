"""CPCV tests: group partition, combinatorics (C(N,k) folds, phi paths), and the no-leakage guard
(train never intersects a test group and respects the purge+embargo gap on BOTH sides)."""
from __future__ import annotations

import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_benchmark.backtest.cpcv import assemble_paths, cpcv_splits, make_groups, n_paths  # noqa: E402


def test_make_groups_partition_is_exact_and_contiguous():
    groups = make_groups(1000, 7)
    assert len(groups) == 7
    assert groups[0][0] == 0 and groups[-1][1] == 1000
    for (a0, a1), (b0, b1) in zip(groups, groups[1:]):
        assert a1 == b0                       # contiguous, no gaps / overlaps
        assert a1 > a0                        # non-empty
    sizes = [b - a for a, b in groups]
    assert max(sizes) - min(sizes) <= 1       # near-equal


def test_make_groups_rejects_degenerate():
    for bad in (1, 0):
        try:
            make_groups(1000, bad); assert False
        except ValueError:
            pass
    try:
        make_groups(5, 6); assert False        # more groups than samples
    except ValueError:
        pass


def test_fold_count_and_path_count():
    N, k = 8, 2
    splits = cpcv_splits(4000, n_groups=N, k_test=k, horizon=1)
    assert len(splits) == comb(N, k) == 28
    assert n_paths(N, k) == comb(N - 1, k - 1) == 7
    # k=3 case
    assert len(cpcv_splits(4000, n_groups=6, k_test=3, horizon=1)) == comb(6, 3) == 20
    assert n_paths(6, 3) == comb(5, 2) == 10


def test_no_train_test_overlap_and_purge_gap():
    N, k, H, emb = 8, 2, 5, 5
    n = 4000
    splits = cpcv_splits(n, n_groups=N, k_test=k, horizon=H, embargo=emb)
    for s in splits:
        train = set(s.train_idx.tolist())
        for ts, te in s.test_slices:
            # 1) no overlap
            assert train.isdisjoint(range(ts, te))
            # 2) left purge: no train label window [i, i+H] reaches into [ts, te) -> no train in [ts-H, ts)
            assert train.isdisjoint(range(max(0, ts - H), ts))
            # 3) right embargo: no train in [te, te+emb)
            assert train.isdisjoint(range(te, min(n, te + emb)))
        # train indices are sorted & unique
        assert list(s.train_idx) == sorted(set(s.train_idx.tolist()))


def test_interior_test_group_keeps_train_on_both_sides():
    # a middle test group must leave usable train data BEFORE and AFTER it (straddle property)
    splits = cpcv_splits(8000, n_groups=8, k_test=1, horizon=2, embargo=2)
    mid = next(s for s in splits if s.test_groups == (3,))
    ts, te = mid.test_slices[0]
    assert (mid.train_idx < ts - 2).any() and (mid.train_idx > te + 2).any()


def test_assemble_paths_covers_every_group_once_and_partitions_predictions():
    N, k = 8, 2
    splits = cpcv_splits(4000, n_groups=N, k_test=k, horizon=1)
    paths = assemble_paths(splits, n_groups=N, k_test=k)
    phi = n_paths(N, k)
    assert len(paths) == phi
    seen_pairs = set()
    for path in paths:
        groups_in_path = [g for g, _ in path]
        assert groups_in_path == list(range(N))          # every group once, in time order
        for g, combo_id in path:
            assert g in splits[combo_id].test_groups      # the combo really tests that group
            seen_pairs.add((combo_id, g))
    # the phi paths use each (combo, test-group) prediction EXACTLY once -> perfect cover
    total = sum(len(s.test_groups) for s in splits)
    assert len(seen_pairs) == total == comb(N, k) * k


def test_horizon_zero_rejected_and_embargo_default_equals_horizon():
    try:
        cpcv_splits(1000, n_groups=5, k_test=2, horizon=0); assert False
    except ValueError:
        pass
    # default embargo == horizon: a train index must avoid [te, te+H)
    H = 7
    s = cpcv_splits(3000, n_groups=6, k_test=1, horizon=H)[0]   # tests group 0 -> right side only
    ts, te = s.test_slices[0]
    assert set(s.train_idx.tolist()).isdisjoint(range(te, te + H))


def test_deterministic():
    a = cpcv_splits(5000, n_groups=7, k_test=2, horizon=3)
    b = cpcv_splits(5000, n_groups=7, k_test=2, horizon=3)
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert sa.test_groups == sb.test_groups
        assert np.array_equal(sa.train_idx, sb.train_idx)
