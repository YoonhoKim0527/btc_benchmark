"""Benchmark runner: drives the walk-forward, gates the strategy, scores it, appends the leaderboard.

The referee depends ONLY on benchmark-owned audited internals (backtester, cost model, walk-forward,
random baseline) -- never on participant code. Costs/splits are constants here; submissions cannot
change them. The sealed holdout is structurally firewalled: dev runs never pass holdout indexes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.backtester import run_backtest
from ..backtest.cost_model import CostConfig
from ..backtest.walk_forward import WalkForwardConfig, generate_splits
from ..decisions.random_baseline import count_changes, random_turnover_matched_positions
from ..utils.logging import get_logger
from .contract import BenchmarkData, Strategy
from .validity import run_gates

log = get_logger(__name__)

BENCHMARK_VERSION = "1.0.0"
PPY = 8760
ALL_IN_BPS = 10.0
SPLIT = {"train_months": 24, "val_months": 3, "test_months": 3, "step_months": 3,
         "embargo_bars": None, "purge_overlapping_labels": True, "sealed_holdout_months": 6}


def fast_net_return(close: np.ndarray, p: np.ndarray, all_in_bps: float) -> float:
    """Vectorized Mode-A total net return mirroring run_backtest close_to_close_reference
    (verified bit-equal in tests/test_benchmark_contract.py)."""
    p = np.asarray(p, dtype="float64")
    if len(p) < 2:
        return float("nan")
    ret = close[1:] / close[:-1] - 1.0
    pos = p[:-1]
    prev = np.concatenate([[0.0], p[:-2]])
    turn = np.abs(pos - prev)
    net = pos * ret - turn * (all_in_bps / 10000.0)
    return float(np.prod(1.0 + net) - 1.0)


def _holding_stats(p: np.ndarray) -> dict:
    trades, i, n = [], 0, len(p)
    while i < n:
        if p[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and p[j + 1] == p[i]:
            j += 1
        trades.append(j + 1 - i)
        i = j + 1
    h = np.asarray(trades, dtype="float64")
    return {"n_trades_runs": int(len(h)), "exposure": float(np.mean(p != 0)) if n else np.nan,
            "hold_mean": float(h.mean()) if len(h) else np.nan,
            "hold_median": float(np.median(h)) if len(h) else np.nan}


def _per_year(net: np.ndarray, ts: pd.Series) -> dict:
    yr = pd.to_datetime(ts, utc=True).dt.year.to_numpy()[: len(net)]
    out = {}
    for y in sorted(set(yr.tolist())):
        r = net[yr == y]
        if len(r) < 100:
            continue
        sd = r.std(ddof=1)
        eq = np.cumprod(1 + r)
        out[int(y)] = {"net": round(float(eq[-1] - 1), 4),
                       "sharpe": round(float(r.mean() / sd * np.sqrt(PPY)), 3) if sd > 0 else None,
                       "dd": round(float((eq / np.maximum.accumulate(eq) - 1).min()), 4)}
    return out


def run_benchmark(strategy: Strategy, data: BenchmarkData, *, split_cfg: dict | None = None,
                  gates: bool = True, leaderboard_path: str | Path | None = None,
                  team: str = "anonymous") -> dict:
    """Evaluate one strategy submission. Returns the report dict (and appends the leaderboard)."""
    horizon = max(1, int(getattr(strategy, "horizon", 1)))
    wf = WalkForwardConfig.from_dict({**(split_cfg or SPLIT), "horizon_bars": horizon})
    splits, holdout = generate_splits(data.candles["timestamp_open"], wf)
    if not splits:
        raise ValueError("no walk-forward folds (data too short for the split config)")
    holdout_start = holdout.get("start_idx") or len(data.candles)

    blocks: list[tuple[int, int, np.ndarray]] = []
    fold_gates: list[dict] = []
    for s in splits:
        tr0, tr1 = s.train_range
        te0, te1 = s.test_range
        assert te1 <= holdout_start, "holdout firewall violated by split generation"
        strategy.fit(data, tr0, tr1)
        pos = np.asarray(strategy.positions(data, te0, te1), dtype="float64")
        if pos.shape != (te1 - te0,):
            raise ValueError(f"strategy returned {pos.shape}, expected ({te1 - te0},) for fold {s.split_id}")
        finite = pos[np.isfinite(pos)]
        if finite.size and np.abs(finite).max() > 1.0 + 1e-9:
            raise ValueError("positions must lie in [-1, 1]")
        # gate EVERY fold's test window against the EXACT scored array `pos` (a strategy cannot be
        # causal on one fold and cheat elsewhere, nor cheat on the scored call and be honest on the
        # gate's recomputed calls)
        if gates:
            fold_gates.append({"fold": int(s.split_id),
                               **run_gates(strategy, data, start=te0, end=te1, scored=pos)})
        blocks.append((te0, te1, pos))

    # consolidate (test windows tile; keep first on overlap), positions aligned to candle rows
    first, last = blocks[0][0], max(b[1] for b in blocks)
    pos_full = np.zeros(last - first, dtype="float64")
    seen = np.zeros(last - first, dtype=bool)
    for te0, te1, p in blocks:
        sl = slice(te0 - first, te1 - first)
        keep = ~seen[sl]
        pos_full[sl] = np.where(keep, p, pos_full[sl])
        seen[sl] = True
    cons = data.candles.iloc[first:last].reset_index(drop=True)
    pos_full = np.where(np.isfinite(pos_full), pos_full, 0.0)

    gate_rep: dict
    if gates:
        failed = [g for g in fold_gates if not g.get("passed", False)]
        gate_rep = {"scope": "all_folds", "n_folds_gated": len(fold_gates),
                    "passed": len(failed) == 0,
                    "failed_folds": [g["fold"] for g in failed],
                    "determinism": all(g["determinism"] for g in fold_gates),
                    "future_perturbation": all(g["future_perturbation"] for g in fold_gates),
                    "prefix_invariance": all(g["prefix_invariance"] for g in fold_gates),
                    "per_fold": fold_gates}
    else:
        gate_rep = {"skipped": True, "passed": True}
    disqualified = bool(gates and not gate_rep.get("passed", False))

    res = run_backtest(cons, pos_full, CostConfig(fee_bps=ALL_IN_BPS), periods_per_year=PPY)
    met = res.metrics(periods_per_year=PPY)
    close = pd.to_numeric(cons["close"], errors="coerce").to_numpy("float64")
    net1x = float(met["total_net_return"])
    costs = {f"net_cost{m:g}x": round(fast_net_return(close, pos_full, m * ALL_IN_BPS), 4)
             for m in (0.0, 1.0, 2.0, 3.0, 5.0)}
    nc = count_changes(pos_full)
    rnd_pct = None
    if nc >= 2:
        rnd = random_turnover_matched_positions(len(pos_full), nc, n_trials=100, seed=42)
        nets = np.array([fast_net_return(close, rp, ALL_IN_BPS) for rp in rnd])
        rnd_pct = round(float(np.mean(nets < net1x)), 3)
    res_no = run_backtest(cons, pos_full, CostConfig(fee_bps=ALL_IN_BPS),
                          execution_mode="next_open_conservative", periods_per_year=PPY)
    fund_net = None
    if "funding" in data.aux and len(data.aux["funding"]):
        res_f = run_backtest(cons, pos_full, CostConfig(fee_bps=ALL_IN_BPS, funding_enabled=True),
                             funding=data.aux["funding"], periods_per_year=PPY)
        fund_net = round(float(res_f.metrics(periods_per_year=PPY)["total_net_return"]), 4)

    report = {
        "benchmark_version": BENCHMARK_VERSION, "team": team, "strategy": strategy.name,
        "declared_horizon": horizon, "n_folds": len(splits),
        "oos_start": str(pd.to_datetime(cons["timestamp_open"].iloc[0])),
        "oos_end": str(pd.to_datetime(cons["timestamp_open"].iloc[-1])),
        "gates": gate_rep, "disqualified": disqualified,
        "net": round(net1x, 4), "sharpe": round(float(met["sharpe"]), 3),
        "sortino": round(float(met.get("sortino", np.nan)), 3),
        "max_drawdown": round(float(met["max_drawdown"]), 4),
        "profit_factor": round(float(met.get("profit_factor", np.nan)), 3),
        "n_trades": int(met["n_trades"]), "turnover": float(met["total_turnover"]),
        **_holding_stats(pos_full), **costs,
        "net_next_open": round(float(res_no.metrics(periods_per_year=PPY)["total_net_return"]), 4),
        "net_funding_aware": fund_net, "random_pctile": rnd_pct,
        "per_year": _per_year(np.asarray(res.net_returns, "float64"), cons["timestamp_open"]),
        "buy_hold_net": round(float(close[-1] / close[0] - 1.0), 4) if len(close) > 1 else None,
        "sealed_holdout_used": False,
    }
    if leaderboard_path is not None:
        p = Path(leaderboard_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(report, default=str) + "\n")
    log.info("benchmark %s: sharpe=%.3f net=%.1f%% dq=%s", strategy.name, report["sharpe"],
             100 * report["net"], disqualified)
    return report
