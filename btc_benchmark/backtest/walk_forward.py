"""Walk-forward splitter with label-purging, embargo, and a sealed final holdout.

Temporal integrity (no random splits): train strictly before validation strictly before test.
A final sealed holdout (last `sealed_holdout_months`) is carved off and NEVER appears in any
fold -- it is reserved for one-shot final evaluation later.

Leakage control (López de Prado purge + embargo), index-exact:
  - A fixed-horizon label at bar i uses close_{i+H}. A TRAIN sample whose label endpoint (i+H)
    reaches the validation region leaks -> purge it. Same for VAL samples reaching the test region.
  - Embargo additionally drops a buffer of `embargo_bars` bars before the next region.
  - Combined gap before the next region = max(H if purge else 0, embargo_bars).
  - Test labels are capped so they cannot read into the sealed holdout (holdout stays pristine).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

_MAX_FOLDS = 10_000  # runaway guard


@dataclass
class WalkForwardConfig:
    train_months: int = 24
    val_months: int = 3
    test_months: int = 3
    step_months: int = 3
    horizon_bars: int = 1
    embargo_bars: int | None = None          # default -> horizon_bars
    purge_overlapping_labels: bool = True
    sealed_holdout_months: int = 6

    def resolved_embargo(self) -> int:
        return self.embargo_bars if self.embargo_bars is not None else self.horizon_bars

    @classmethod
    def from_dict(cls, d: dict) -> "WalkForwardConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Split:
    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_range: tuple[int, int]   # [start, stop) AFTER purge/embargo
    val_range: tuple[int, int]
    test_range: tuple[int, int]
    train_n_before: int
    val_n_before: int
    test_n_before: int
    train_n_after: int
    val_n_after: int
    test_n_after: int
    embargo_bars: int
    horizon_bars: int
    sealed_holdout_start: Any = None
    sealed_holdout_end: Any = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end",
                  "sealed_holdout_start", "sealed_holdout_end"):
            d[k] = str(d[k]) if d[k] is not None else None
        return d


def generate_splits(timestamps, config: WalkForwardConfig) -> tuple[list[Split], dict]:
    """Return (splits, holdout_info). `timestamps` must be sorted ascending and unique."""
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted ascending")
    if ts.has_duplicates:
        raise ValueError("timestamps must be unique")
    M = len(ts)
    if M == 0:
        return [], {"start": None, "end": None, "start_idx": 0, "n": 0, "months": config.sealed_holdout_months}
    t0, tN = ts[0], ts[-1]
    H = config.horizon_bars
    emb = config.resolved_embargo()
    gap = max(H if config.purge_overlapping_labels else 0, emb)

    holdout_start_time = tN - pd.DateOffset(months=config.sealed_holdout_months)
    holdout_start_idx = int(ts.searchsorted(holdout_start_time, side="left"))
    holdout_start_idx = max(0, min(holdout_start_idx, M))
    has_holdout = 0 < holdout_start_idx < M
    sealed_start = ts[holdout_start_idx] if has_holdout else None
    sealed_end = tN if has_holdout else None
    # test labels must not read into the holdout
    dev_label_cap = holdout_start_idx - (H if config.purge_overlapping_labels else 0)

    splits: list[Split] = []
    for k in range(_MAX_FOLDS):
        train_start = t0 + pd.DateOffset(months=config.step_months * k)
        train_end = train_start + pd.DateOffset(months=config.train_months)
        val_start = train_end
        val_end = val_start + pd.DateOffset(months=config.val_months)
        test_start = val_end
        test_end = test_start + pd.DateOffset(months=config.test_months)
        if test_end > holdout_start_time:
            break
        a = int(ts.searchsorted(train_start, side="left"))
        vs = int(ts.searchsorted(val_start, side="left"))
        te_start = int(ts.searchsorted(test_start, side="left"))
        te_end = min(int(ts.searchsorted(test_end, side="left")), dev_label_cap)
        train_n_before = vs - a
        val_n_before = te_start - vs
        test_n_before = te_end - te_start
        if train_n_before <= 0 or val_n_before <= 0 or test_n_before <= 0:
            continue
        train_stop = max(a, vs - gap)
        val_stop = max(vs, te_start - gap)
        splits.append(Split(
            split_id=len(splits),
            train_start=train_start, train_end=train_end, val_start=val_start, val_end=val_end,
            test_start=test_start, test_end=test_end,
            train_range=(a, train_stop), val_range=(vs, val_stop), test_range=(te_start, te_end),
            train_n_before=train_n_before, val_n_before=val_n_before, test_n_before=test_n_before,
            train_n_after=train_stop - a, val_n_after=val_stop - vs, test_n_after=te_end - te_start,
            embargo_bars=emb, horizon_bars=H,
            sealed_holdout_start=sealed_start, sealed_holdout_end=sealed_end,
        ))
    holdout = {
        "start": str(sealed_start) if sealed_start is not None else None,
        "end": str(sealed_end) if sealed_end is not None else None,
        "start_idx": int(holdout_start_idx) if has_holdout else None,
        "n": int(M - holdout_start_idx) if has_holdout else 0,
        "months": config.sealed_holdout_months,
        "used": False,  # set True only when an experiment explicitly evaluates on it
    }
    return splits, holdout


def holdout_range(timestamps, config: WalkForwardConfig) -> tuple[int, int] | None:
    """Index range [start, len) of the sealed holdout, or None if absent."""
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    M = len(ts)
    hs = int(ts.searchsorted(ts[-1] - pd.DateOffset(months=config.sealed_holdout_months), side="left"))
    return (hs, M) if 0 < hs < M else None


def summarize(splits: list[Split], holdout: dict) -> dict:
    return {
        "n_folds": len(splits),
        "first_train_start": splits[0].to_dict()["train_start"] if splits else None,
        "last_test_end": splits[-1].to_dict()["test_end"] if splits else None,
        "sealed_holdout": holdout,
        "folds": [s.to_dict() for s in splits],
    }
