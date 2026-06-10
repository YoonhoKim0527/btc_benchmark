"""Performance metrics computed from backtest per-period series + trade log.

Robustness: zero-trade / all-cash / zero-volatility cases return NaN (with a `notes` entry),
never a crash or a silently-good infinity.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8760
_STD_EPS = 1e-12  # below this, volatility is treated as zero (degenerate series) -> Sharpe NaN


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1.0
    return float(dd.min())  # <= 0


def compute_metrics(
    net_returns,
    *,
    equity=None,
    positions=None,
    gross_returns=None,
    costs=None,
    funding=None,
    trades: pd.DataFrame | None = None,
    periods_per_year: int = HOURS_PER_YEAR,
    risk_free: float = 0.0,
) -> dict[str, Any]:
    net = np.asarray(net_returns, dtype="float64")
    notes: list[str] = []
    n = len(net)
    if n == 0:
        return {"n_periods": 0, "notes": "no periods"}
    if equity is None:
        equity = np.cumprod(1.0 + net)
    equity = np.asarray(equity, dtype="float64")

    total_net = float(equity[-1] - 1.0)
    total_gross = float(np.prod(1.0 + np.asarray(gross_returns, dtype="float64")) - 1.0) if gross_returns is not None else float("nan")
    years = n / periods_per_year
    # annualize in log-space with an overflow guard (annualizing very short backtests can blow up)
    if years > 0 and (1.0 + total_net) > 0:
        log_growth = np.log(1.0 + total_net) / years
        arc = float(np.expm1(log_growth)) if log_growth < 700 else float("inf")
        if log_growth >= 700:
            notes.append("annualized_return overflow (backtest too short to annualize)")
    else:
        arc = float("nan")
    asd = float(np.std(net, ddof=1) * np.sqrt(periods_per_year)) if n > 1 else float("nan")

    mean = float(np.mean(net))
    std = float(np.std(net, ddof=1)) if n > 1 else float("nan")
    # near-zero volatility (e.g. a constant series, whose float std is ~1e-19) must NOT yield a
    # giant "good" Sharpe -> report NaN with a reason instead.
    if std and std > _STD_EPS and not np.isnan(std):
        sharpe = (mean - risk_free / periods_per_year) / std * np.sqrt(periods_per_year)
    else:
        sharpe = float("nan"); notes.append("sharpe NaN: zero/near-zero volatility")
    downside = net[net < 0]
    dstd = float(np.std(downside, ddof=1)) if len(downside) > 1 else float("nan")
    sortino = (mean / dstd * np.sqrt(periods_per_year)) if dstd and dstd > _STD_EPS and not np.isnan(dstd) else float("nan")
    md = _max_drawdown(equity)
    calmar = (arc / abs(md)) if md and md < 0 and np.isfinite(arc) else float("nan")  # isfinite blocks NaN+inf

    # exposure
    if positions is not None:
        p = np.asarray(positions, dtype="float64")
        long_frac = float(np.mean(p > 0)); short_frac = float(np.mean(p < 0)); cash_frac = float(np.mean(p == 0))
    else:
        long_frac = short_frac = cash_frac = float("nan")

    total_turnover = float(np.sum(np.abs(np.diff(np.concatenate([[0.0], np.asarray(positions, "float64")]))))) if positions is not None else float("nan")
    cost_total = float(np.sum(costs)) if costs is not None else float("nan")
    funding_total = float(np.nansum(funding)) if funding is not None else float("nan")
    gross_pnl = total_gross if not np.isnan(total_gross) else float("nan")
    cost_pct_of_gross = (cost_total / abs(gross_pnl)) if (not np.isnan(gross_pnl) and gross_pnl != 0 and not np.isnan(cost_total)) else float("nan")

    # trade-level
    if trades is not None and len(trades):
        tr = trades["net_return"].to_numpy(dtype="float64")
        wins = tr[tr > 0]; losses = tr[tr < 0]
        win_rate = float(len(wins) / len(tr))
        avg_win = float(np.mean(wins)) if len(wins) else float("nan")
        avg_loss = float(np.mean(losses)) if len(losses) else float("nan")
        payoff = (avg_win / abs(avg_loss)) if (len(wins) and len(losses)) else float("nan")
        profit_factor = (float(wins.sum() / abs(losses.sum()))) if (len(wins) and losses.sum() != 0) else float("nan")
        n_trades = int(len(tr))
        avg_holding = float(trades["holding_period_bars"].mean())
        pos_pnl = tr[tr > 0]
        top5 = float(np.sort(pos_pnl)[::-1][:5].sum()) if len(pos_pnl) else 0.0
        profit_concentration = (top5 / pos_pnl.sum()) if pos_pnl.sum() > 0 else float("nan")
    else:
        win_rate = avg_win = avg_loss = payoff = profit_factor = avg_holding = profit_concentration = float("nan")
        n_trades = 0
        if trades is not None:
            notes.append("no trades")

    return {
        "n_periods": n,
        "total_net_return": total_net,
        "total_gross_return": total_gross,
        "annualized_return": arc,
        "annualized_volatility": asd,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": md,
        "calmar": calmar,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "profit_factor": profit_factor,
        "total_turnover": total_turnover,
        "n_trades": n_trades,
        "avg_holding_period_bars": avg_holding,
        "exposure_long_fraction": long_frac,
        "exposure_short_fraction": short_frac,
        "exposure_cash_fraction": cash_frac,
        "profit_concentration_top5": profit_concentration,
        "cost_paid_total": cost_total,
        "funding_paid_total": funding_total,
        "cost_pct_of_gross_pnl": cost_pct_of_gross,
        "notes": "; ".join(notes) if notes else "",
    }
