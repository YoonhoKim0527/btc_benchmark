"""Benchmark <-> agent-system contract (v1). See docs/BENCHMARK_SPLIT_DESIGN.md.

The benchmark DRIVES the walk-forward and calls the strategy per fold. A strategy:
  - fit(data, train_start, train_end): may use ONLY rows [train_start, train_end)
  - positions(data, start, end): returns p[t] in [-1, 1] for t in [start, end); p[t] may use only
    information up to bar t (candle rows <= t; aux events with event_time <= close[t]).
  - horizon: declared label horizon in bars (the benchmark purges/embargoes at this horizon and
    prints it in the report; deliberately under-declaring is the one dishonesty the gates cannot
    see, so it is surfaced, versioned, and auditable).
Causal use within a window is NOT trusted -- it is enforced by the gates in validity.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import pandas as pd


class BenchmarkData(NamedTuple):
    candles: pd.DataFrame                  # canonical 1h frame (timestamps, OHLCV, is_imputed)
    aux: dict[str, pd.DataFrame]           # raw event/sub-bar frames (may be empty)


@runtime_checkable
class Strategy(Protocol):
    name: str
    horizon: int

    def fit(self, data: BenchmarkData, train_start: int, train_end: int) -> None: ...

    def positions(self, data: BenchmarkData, start: int, end: int) -> np.ndarray: ...


_AUX_GLOBS = {
    "funding": "data/raw/binance/futures_um/fundingRate/BTCUSDT/*funding.parquet",
    "open_interest": "data/raw/binance/futures_um/metrics/BTCUSDT/*open_interest.parquet",
    "mark_premium": "data/raw/binance/futures_um/markPremium/BTCUSDT/*mark_premium.parquet",
    "sub5": "data/raw/binance/futures_um/klines/BTCUSDT/5m/BTCUSDT_5m_raw.parquet",
    "sub15": "data/raw/binance/futures_um/klines/BTCUSDT/15m/BTCUSDT_15m_raw.parquet",
    "sub1": "data/raw/binance/futures_um/klines/BTCUSDT/1m/BTCUSDT_1m_raw.parquet",
}


def load_benchmark_data(repo_root: str | Path = ".", *, include_aux: bool = True,
                        include_sub_bars: bool = True) -> BenchmarkData:
    """Load the canonical data bundle from a benchmark-repo checkout."""
    root = Path(repo_root)
    candles = pd.read_parquet(root / "data/processed/BTCUSDT_futures_um_1h.parquet")
    aux: dict[str, pd.DataFrame] = {}
    if include_aux:
        import glob as _g
        for k, pat in _AUX_GLOBS.items():
            if not include_sub_bars and k.startswith("sub"):
                continue
            hits = _g.glob(str(root / pat))
            if hits:
                aux[k] = pd.read_parquet(hits[0])
    return BenchmarkData(candles=candles, aux=aux)
