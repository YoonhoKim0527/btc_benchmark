"""Backtester core. Correct accounting first; returns are not optimized here.

Execution modes (recorded in the result):
  - close_to_close_reference (Mode A, default): position p_t decided at completed bar t earns
    the return close_t -> close_{t+1}; cost of entering p_t (|p_t - p_{t-1}|) is charged in that
    same period. Reproduces the common research-paper convention (and the reference paper eq. 4).
  - next_open_conservative (Mode B): p_t decided at close_t is executed at open_{t+1} and earns
    open_{t+1} -> open_{t+2}. More implementable / conservative. Implemented but less battle-tested.

No look-ahead: p_t uses only information up to bar t. The last position is never "held" (no
future bar), so it earns nothing and its entry cost is not charged.

Funding sign (long_pays_when_positive): a position of direction d over a funding event with rate
f contributes -d * f to that period's return (long pays when f > 0; short receives).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.io import git_commit_hash, read_parquet, utcnow_iso, write_manifest
from ..utils.logging import get_logger
from . import metrics as _metrics
from .cost_model import CostConfig

log = get_logger(__name__)

EXECUTION_MODES = ("close_to_close_reference", "next_open_conservative")
_TRADE_COLUMNS = [
    "trade_id", "entry_time", "exit_time", "side", "entry_price", "exit_price",
    "gross_return", "net_return", "cost_paid", "funding_paid", "holding_period_bars",
    "exit_reason", "touched_imputed_candle",
]


@dataclass
class BacktestResult:
    period_start: pd.DatetimeIndex
    positions: np.ndarray
    gross_returns: np.ndarray
    costs: np.ndarray
    funding: np.ndarray
    net_returns: np.ndarray
    turnover: np.ndarray
    equity: np.ndarray
    trades: pd.DataFrame
    execution_mode: str
    meta: dict = field(default_factory=dict)

    def metrics(self, *, periods_per_year: int = 8760, risk_free: float = 0.0) -> dict[str, Any]:
        return _metrics.compute_metrics(
            self.net_returns, equity=self.equity, positions=self.positions,
            gross_returns=self.gross_returns, costs=self.costs, funding=self.funding,
            trades=self.trades, periods_per_year=periods_per_year, risk_free=risk_free)


def _funding_per_period(funding, bnd_ns: np.ndarray, pos_period: np.ndarray, sign: str) -> np.ndarray:
    """Assign each funding event to the period whose holding interval (bnd[t], bnd[t+1]] contains
    it, and accumulate -d*rate (long_pays_when_positive). Vectorized."""
    out = np.zeros(len(pos_period), dtype="float64")
    if funding is None or len(funding) == 0:
        return out
    fe = pd.DatetimeIndex(pd.to_datetime(funding["event_time"], utc=True)).as_unit("ns").asi8
    rate = pd.to_numeric(funding["funding_rate"], errors="coerce").to_numpy(dtype="float64")
    idx = np.searchsorted(bnd_ns, fe, side="left")  # event in (bnd[idx-1], bnd[idx]]
    period = idx - 1
    valid = (idx >= 1) & (idx <= len(bnd_ns) - 1) & ~np.isnan(rate)
    if sign != "long_pays_when_positive":
        raise ValueError(f"unknown funding_sign {sign!r}")
    contrib = -pos_period[period[valid]] * rate[valid]
    np.add.at(out, period[valid], contrib)
    return out


def _build_trade_log(pos, entry_px, exit_px, gross, net, cost, fund, hold_start, hold_end,
                     imp_touch) -> pd.DataFrame:
    rows = []
    M = len(pos)
    i = 0
    tid = 0
    while i < M:
        if pos[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < M and pos[j + 1] == pos[i]:
            j += 1
        sl = slice(i, j + 1)
        rows.append({
            "trade_id": tid,
            "entry_time": hold_start[i],
            "exit_time": hold_end[j],
            "side": "long" if pos[i] > 0 else "short",
            "entry_price": float(entry_px[i]),
            "exit_price": float(exit_px[j]),
            "gross_return": float(np.prod(1.0 + gross[sl]) - 1.0),
            "net_return": float(np.prod(1.0 + net[sl]) - 1.0),
            "cost_paid": float(np.sum(cost[sl])),
            "funding_paid": float(np.sum(fund[sl])),
            "holding_period_bars": int(j - i + 1),
            "exit_reason": "signal" if j + 1 < M else "end_of_data",
            "touched_imputed_candle": bool(np.any(imp_touch[sl])),
        })
        tid += 1
        i = j + 1
    if not rows:
        return pd.DataFrame(columns=_TRADE_COLUMNS)
    return pd.DataFrame(rows)[_TRADE_COLUMNS]


def run_backtest(candles: pd.DataFrame, positions, cost_config: CostConfig, *,
                 execution_mode: str = "close_to_close_reference",
                 funding=None, periods_per_year: int = 8760) -> BacktestResult:
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(f"execution_mode must be one of {EXECUTION_MODES}")
    c = candles.reset_index(drop=True)
    close = c["close"].to_numpy(dtype="float64")
    N = len(close)
    # prices must be finite -- a NaN/Inf close would silently NaN the whole equity curve
    if not np.isfinite(close).all():
        raise ValueError("close contains non-finite values (NaN/Inf) -- fix upstream; do not silently backtest")
    if execution_mode == "close_to_close_reference" and N < 2:
        raise ValueError("close_to_close_reference needs >= 2 bars")
    if execution_mode == "next_open_conservative" and N < 3:
        raise ValueError("next_open_conservative needs >= 3 bars")
    p = np.asarray(positions, dtype="float64")
    if len(p) != N:
        raise ValueError(f"positions length {len(p)} != candles length {N}")
    if not np.isfinite(p).all():
        raise ValueError("positions contain non-finite values")
    is_imp = (pd.Series(c["is_imputed"]).astype("boolean").fillna(False).astype(bool).to_numpy()
              if "is_imputed" in c.columns else np.zeros(N, dtype=bool))
    ts_open = pd.to_datetime(c["timestamp_open"], utc=True)
    # funding bucketing (searchsorted) assumes time order; enforce it rather than silently mis-bucket
    if not ts_open.is_monotonic_increasing:
        raise ValueError("candles must be sorted ascending by timestamp_open")
    if ts_open.duplicated().any():
        raise ValueError("candles contain duplicate timestamp_open")
    ts_close = (pd.to_datetime(c["timestamp_close"], utc=True) if "timestamp_close" in c.columns else ts_open)

    prev_full = np.concatenate([[0.0], p[:-1]])  # p[t-1]
    turn_full = np.abs(p - prev_full)            # entering p[t]

    if execution_mode == "close_to_close_reference":
        ret = close[1:] / close[:-1] - 1.0
        pos_period = p[:-1]
        turnover = turn_full[:-1]
        entry_px = close[:-1]
        exit_px = close[1:]
        hold_start = ts_close.to_numpy()[:-1]    # (close_t, close_{t+1}]
        hold_end = ts_close.to_numpy()[1:]
        bnd_ns = pd.DatetimeIndex(ts_close).as_unit("ns").asi8        # length N -> M+1
        period_start = ts_close.iloc[:-1]
        imp_touch = is_imp[:-1] | is_imp[1:]
    else:  # next_open_conservative
        if "open" not in c.columns:
            raise KeyError("next_open_conservative requires an 'open' column")
        op = c["open"].to_numpy(dtype="float64")
        if not np.isfinite(op).all():
            raise ValueError("open contains non-finite values (NaN/Inf)")
        ret = op[2:] / op[1:-1] - 1.0
        pos_period = p[:-2]
        turnover = turn_full[:-2]
        entry_px = op[1:-1]
        exit_px = op[2:]
        hold_start = ts_open.to_numpy()[1:-1]     # (open_{t+1}, open_{t+2}]
        hold_end = ts_open.to_numpy()[2:]
        bnd_ns = pd.DatetimeIndex(ts_open).as_unit("ns").asi8[1:]     # length N-1 -> M+1
        period_start = ts_open.iloc[1:-1]
        imp_touch = is_imp[1:-1] | is_imp[2:]

    M = len(ret)
    gross = pos_period * ret
    cost = turnover * cost_config.all_in_cost_bps / 10000.0
    fund = np.zeros(M, dtype="float64")
    if cost_config.funding_enabled:
        fund = _funding_per_period(funding, bnd_ns, pos_period, cost_config.funding_sign)
    net = gross - cost + fund
    equity = np.cumprod(1.0 + net) if M else np.array([], dtype="float64")
    trades = _build_trade_log(pos_period, entry_px, exit_px, gross, net, cost, fund,
                              hold_start, hold_end, imp_touch)
    meta = {
        "execution_mode": execution_mode,
        "n_bars": int(N),
        "n_periods": int(M),
        "all_in_cost_bps": cost_config.all_in_cost_bps,
        "funding_enabled": bool(cost_config.funding_enabled),
        "funding_applied": bool(cost_config.funding_enabled and funding is not None and len(funding)),
        "periods_per_year": periods_per_year,
        "git_commit": git_commit_hash(),
        "generated_at": utcnow_iso(),
    }
    return BacktestResult(period_start.reset_index(drop=True), pos_period, gross, cost, fund,
                          net, turnover, equity, trades, execution_mode, meta)


def cost_sensitivity(candles, positions, cost_config, *, multipliers=(0.0, 0.5, 1.0, 2.0, 3.0),
                     execution_mode="close_to_close_reference", funding=None, periods_per_year=8760):
    """Run the same positions across cost multipliers; return {mult: metrics}."""
    out = {}
    for m in multipliers:
        res = run_backtest(candles, positions, cost_config.scaled(m), execution_mode=execution_mode,
                           funding=funding, periods_per_year=periods_per_year)
        out[m] = res.metrics(periods_per_year=periods_per_year)
    return out
