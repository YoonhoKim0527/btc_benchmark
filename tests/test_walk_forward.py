"""Milestone 4 tests: walk-forward splitter (ordering, purge/embargo, sealed holdout)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_benchmark.backtest.walk_forward import WalkForwardConfig, generate_splits  # noqa: E402


def hourly(months: int, start="2018-01-01"):
    end = pd.Timestamp(start, tz="UTC") + pd.DateOffset(months=months)
    return pd.date_range(start=pd.Timestamp(start, tz="UTC"), end=end, freq="1h", inclusive="left")


CFG = WalkForwardConfig(train_months=6, val_months=1, test_months=1, step_months=1,
                        horizon_bars=1, sealed_holdout_months=2)


def test_generates_multiple_folds():
    splits, holdout = generate_splits(hourly(18), CFG)
    assert len(splits) >= 3
    assert holdout["start_idx"] is not None and holdout["n"] > 0
    assert holdout["used"] is False


def test_ordering_and_no_overlap():
    ts = hourly(18)
    splits, _ = generate_splits(ts, CFG)
    for s in splits:
        assert s.train_range[0] < s.train_range[1] <= s.val_range[0]
        assert s.val_range[0] < s.val_range[1] <= s.test_range[0]
        assert s.test_range[0] < s.test_range[1]
        # strict time ordering across regions
        assert ts[s.train_range[1] - 1] < ts[s.val_range[0]]
        assert ts[s.val_range[1] - 1] < ts[s.test_range[0]]


def test_label_endpoints_do_not_cross_boundaries_H1():
    ts = hourly(18)
    splits, _ = generate_splits(ts, CFG)
    H = CFG.horizon_bars
    for s in splits:
        # last kept train sample's label endpoint must stay inside train (before val)
        assert (s.train_range[1] - 1) + H < s.val_range[0]
        assert (s.val_range[1] - 1) + H < s.test_range[0]


def test_label_endpoints_do_not_cross_boundaries_H4():
    ts = hourly(18)
    cfg = WalkForwardConfig(train_months=6, val_months=1, test_months=1, step_months=1,
                            horizon_bars=4, embargo_bars=4, sealed_holdout_months=2)
    splits, _ = generate_splits(ts, cfg)
    assert len(splits) >= 1
    for s in splits:
        assert (s.train_range[1] - 1) + 4 < s.val_range[0]
        assert (s.val_range[1] - 1) + 4 < s.test_range[0]
        assert s.train_n_after <= s.train_n_before  # purge/embargo removed samples


def test_sealed_holdout_untouched():
    ts = hourly(18)
    splits, holdout = generate_splits(ts, CFG)
    hs = holdout["start_idx"]
    H = CFG.horizon_bars
    for s in splits:
        # no fold index reaches the holdout, AND no test label reads into it
        assert s.test_range[1] <= hs
        assert (s.test_range[1] - 1) + H < hs


def test_deterministic():
    ts = hourly(18)
    a, _ = generate_splits(ts, CFG)
    b, _ = generate_splits(ts, CFG)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]


def test_requires_sorted_unique():
    ts = hourly(12)
    with pytest.raises(ValueError):
        generate_splits(ts[::-1], CFG)            # unsorted
    dup = ts.append(ts[[5]]).sort_values()
    with pytest.raises(ValueError):
        generate_splits(dup, CFG)                 # duplicate
