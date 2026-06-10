"""Benchmark contract tests: honest strategy scores, cheaters are disqualified by the gates,
the holdout is structurally firewalled, and the referee's fast net path is bit-equal to the
audited backtester."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_benchmark.backtest.backtester import run_backtest  # noqa: E402
from btc_benchmark.backtest.cost_model import CostConfig  # noqa: E402
from btc_benchmark.benchmark import BenchmarkData, load_benchmark_data, run_benchmark  # noqa: E402
from btc_benchmark.benchmark.runner import fast_net_return  # noqa: E402

SPLIT_SMALL = {"train_months": 2, "val_months": 1, "test_months": 1, "step_months": 1,
               "embargo_bars": None, "purge_overlapping_labels": True, "sealed_holdout_months": 1}


def _data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    close = 30000.0 * np.exp(np.cumsum(rng.normal(0.00005, 0.01, n)))
    o = np.concatenate([[close[0]], close[:-1]])
    candles = pd.DataFrame({"timestamp_open": ts,
                            "timestamp_close": ts + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
                            "open": o, "high": np.maximum(o, close) * 1.002,
                            "low": np.minimum(o, close) * 0.998, "close": close,
                            "volume": rng.uniform(50, 500, n), "is_imputed": False})
    fund = pd.DataFrame({"event_time": ts[::8], "funding_rate": rng.normal(1e-4, 2e-4, len(ts[::8])),
                         "funding_interval_hours": 8})
    return BenchmarkData(candles=candles, aux={"funding": fund})


class EmaRule:
    """Honest causal strategy: long when close > trailing EMA(48)."""
    name, horizon = "ema_rule", 1

    def fit(self, data, train_start, train_end):
        self.seen_max = max(getattr(self, "seen_max", 0), train_end)

    def positions(self, data, start, end):
        self.seen_max = max(getattr(self, "seen_max", 0), end)
        close = pd.to_numeric(data.candles["close"].iloc[:end], errors="coerce")
        ema = close.ewm(span=48, adjust=False).mean()
        pos = (close > ema).astype(float).to_numpy()
        return pos[start:end]


class FutureCheater:
    """Uses tomorrow's close -- must be caught by the future-perturbation gate."""
    name, horizon = "future_cheater", 1

    def fit(self, data, train_start, train_end):
        pass

    def positions(self, data, start, end):
        c = pd.to_numeric(data.candles["close"], errors="coerce").to_numpy("float64")
        nxt = np.concatenate([c[1:], [c[-1]]])
        return (nxt[start:end] > c[start:end]).astype("float64")


class EndDependent:
    """Past-data-only but depends on the requested window END -- prefix gate must catch it."""
    name, horizon = "end_dependent", 1

    def fit(self, data, train_start, train_end):
        pass

    def positions(self, data, start, end):
        idx = np.arange(start, end)
        return ((end - idx) % 2 == 0).astype("float64")


def test_honest_strategy_scores_and_passes_gates(tmp_path):
    data = _data()
    rep = run_benchmark(EmaRule(), data, split_cfg=SPLIT_SMALL,
                        leaderboard_path=tmp_path / "lb.jsonl")
    assert rep["gates"]["passed"] and not rep["disqualified"]
    for k in ("net", "sharpe", "max_drawdown", "net_cost2x", "net_next_open",
              "net_funding_aware", "random_pctile", "per_year", "buy_hold_net"):
        assert k in rep, k
    assert rep["sealed_holdout_used"] is False
    assert (tmp_path / "lb.jsonl").exists()


def test_future_cheater_disqualified():
    rep = run_benchmark(FutureCheater(), _data(), split_cfg=SPLIT_SMALL)
    assert rep["gates"]["future_perturbation"] is False
    assert rep["disqualified"] is True


def test_end_dependent_caught_by_prefix_gate():
    rep = run_benchmark(EndDependent(), _data(), split_cfg=SPLIT_SMALL)
    assert rep["gates"]["determinism"] is True
    assert rep["gates"]["future_perturbation"] is True     # no data dependence on the future
    assert rep["gates"]["prefix_invariance"] is False      # but the window end changes decisions
    assert rep["disqualified"] is True


def test_holdout_structurally_firewalled():
    data = _data()
    s = EmaRule()
    run_benchmark(s, data, split_cfg=SPLIT_SMALL)
    from btc_benchmark.backtest.walk_forward import WalkForwardConfig, holdout_range
    hr = holdout_range(data.candles["timestamp_open"],
                       WalkForwardConfig.from_dict({**SPLIT_SMALL, "horizon_bars": 1}))
    assert hr is not None and s.seen_max <= hr[0]          # strategy never saw a holdout index


def test_fast_net_return_bit_equal_to_backtester():
    data = _data(3000)
    rng = np.random.default_rng(3)
    pos = rng.choice([0.0, 1.0, -1.0], size=3000, p=[0.5, 0.3, 0.2])
    close = data.candles["close"].to_numpy("float64")
    fast = fast_net_return(close, pos, 10.0)
    full = run_backtest(data.candles, pos, CostConfig(fee_bps=10.0),
                        periods_per_year=8760).metrics(periods_per_year=8760)["total_net_return"]
    assert fast == full                                     # bit-exact, not approx


def test_load_benchmark_data_real_repo():
    import pytest
    if not Path("data/processed/BTCUSDT_futures_um_1h.parquet").exists():
        pytest.skip("data bundle not built (run scripts.bootstrap_data)")
    data = load_benchmark_data(".", include_sub_bars=False)
    assert len(data.candles) > 50000 and "funding" in data.aux


# ===== audit regressions (C1 fold-selective bypass, H2 sub-bar straddle, M1/M2 accounting) =====
class FoldSelectiveCheater:
    """Causal ONLY on the first test fold; uses close[t+1] look-ahead on every other fold.
    Pre-fix this passed (gates ran on one fold); now every fold is gated -> must be disqualified."""
    name, horizon = "fold_selective_cheater", 1

    def __init__(self):
        from btc_benchmark.backtest.walk_forward import WalkForwardConfig, generate_splits
        self._wf = WalkForwardConfig, generate_splits

    def fit(self, data, train_start, train_end):
        pass

    def positions(self, data, start, end):
        WalkForwardConfig, generate_splits = self._wf
        splits, _ = generate_splits(data.candles["timestamp_open"],
                                    WalkForwardConfig.from_dict({**SPLIT_SMALL, "horizon_bars": 1}))
        honest_fold = splits[0].test_range if splits else (start, end)
        c = pd.to_numeric(data.candles["close"], errors="coerce").to_numpy("float64")
        if (start, end) == tuple(honest_fold):
            ema = pd.Series(c[:end]).ewm(span=24, adjust=False).mean().to_numpy()
            return (c[start:end] > ema[start:end]).astype("float64")
        nxt = np.concatenate([c[1:], [c[-1]]])               # look-ahead on the other folds
        return (nxt[start:end] > c[start:end]).astype("float64")


def test_fold_selective_cheater_now_disqualified():
    rep = run_benchmark(FoldSelectiveCheater(), _data(), split_cfg=SPLIT_SMALL)
    assert rep["gates"]["scope"] == "all_folds"
    assert rep["gates"]["future_perturbation"] is False      # caught on a non-honest fold
    assert rep["disqualified"] is True
    assert len(rep["gates"]["failed_folds"]) >= 1


class SubBarStraddleCheater:
    """Reads the 5m sub-bar that OPENS <= close[t] but CLOSES after it (5 min of future).
    The gate must perturb straddling sub-bars by their close time and catch this."""
    name, horizon = "subbar_straddle", 1

    def fit(self, data, train_start, train_end):
        pass

    def positions(self, data, start, end):
        s5 = data.aux["sub5"]
        so = pd.to_datetime(s5["timestamp_open"], utc=True)
        c = pd.to_datetime(data.candles["timestamp_close"], utc=True)
        out = np.zeros(end - start)
        for k, t in enumerate(range(start, end)):
            cutoff = c.iloc[t]
            # the 5m bar opening in (cutoff-5min, cutoff]: opens <= close[t] but closes after it
            m = (so > cutoff - pd.Timedelta(minutes=5)) & (so <= cutoff)
            if m.any():
                out[k] = 1.0 if float(s5.loc[m, "close"].iloc[-1]) > 0 else 0.0
        return out


def test_subbar_straddle_cheater_disqualified():
    data = _data()
    ts = pd.to_datetime(data.candles["timestamp_open"], utc=True)
    rows = []
    rng = np.random.default_rng(0)
    for t in ts:
        for kk in range(12):
            rows.append({"timestamp_open": t + pd.Timedelta(minutes=5 * kk),
                         "open": 1.0, "high": 1.0, "low": 1.0,
                         "close": float(rng.normal()), "volume": 1.0})
    data = BenchmarkData(candles=data.candles, aux={**data.aux, "sub5": pd.DataFrame(rows)})
    rep = run_benchmark(SubBarStraddleCheater(), data, split_cfg=SPLIT_SMALL)
    assert rep["gates"]["future_perturbation"] is False
    assert rep["disqualified"] is True


def test_trade_log_includes_exit_cost_and_bh_uses_last_bar():
    from btc_benchmark.backtest.cost_model import CostConfig
    n = 60
    ts = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    close = np.linspace(100, 130, n)
    candles = pd.DataFrame({"timestamp_open": ts,
                            "timestamp_close": ts + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
                            "open": close, "high": close, "low": close, "close": close,
                            "volume": 1.0, "is_imputed": False})
    pos = np.zeros(n); pos[10:20] = 1.0                       # one flat round-trip
    res = run_backtest(candles, pos, CostConfig(fee_bps=10.0), periods_per_year=8760)
    tr = res.trades.iloc[0]
    # entry (bar 10) + exit (bar 20) both 1 unit * 10bps -> 20 bps round-trip, now in the trade row
    assert abs(tr["cost_paid"] - 2 * 10.0 / 10000.0) < 1e-12
    assert abs(float(res.costs.sum()) - tr["cost_paid"]) < 1e-12   # whole portfolio = this one trade
