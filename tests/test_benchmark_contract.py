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
    # This strategy ignores the data entirely and depends only on the window-end parity, so
    # prefix-invariance is its reliable catch (asking for a shorter window flips earlier decisions).
    # The future-perturbation probe also uses a unique window per cutoff, but under the *budgeted*
    # (even-stride) sweep the sampled windows can share base's parity and not flag a pure
    # parity-on-end strategy -- which is exactly why prefix-invariance is a separate gate.
    assert rep["gates"]["prefix_invariance"] is False
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
# 3+ folds so that gating only the MIDDLE fold (the pre-fix behavior) would MISS a cheater honest
# only on that fold -- proving the all-fold fix, not a lucky split.
SPLIT_3F = {"train_months": 2, "val_months": 1, "test_months": 1, "step_months": 1,
            "embargo_bars": None, "purge_overlapping_labels": True, "sealed_holdout_months": 1}


class MiddleFoldOnlyHonest:
    """Honest ONLY on the fold a single-middle-fold gate would have sampled; look-ahead elsewhere."""
    name, horizon = "middle_fold_only_honest", 1

    def __init__(self):
        from btc_benchmark.backtest.walk_forward import WalkForwardConfig, generate_splits
        self._wf = WalkForwardConfig, generate_splits

    def fit(self, data, train_start, train_end):
        pass

    def positions(self, data, start, end):
        WFC, gen = self._wf
        splits, _ = gen(data.candles["timestamp_open"], WFC.from_dict({**SPLIT_3F, "horizon_bars": 1}))
        honest = splits[len(splits) // 2].test_range if splits else (start, end)
        c = pd.to_numeric(data.candles["close"], errors="coerce").to_numpy("float64")
        if (start, end) == tuple(honest):
            ema = pd.Series(c[:end]).ewm(span=24, adjust=False).mean().to_numpy()
            return (c[start:end] > ema[start:end]).astype("float64")
        nxt = np.concatenate([c[1:], [c[-1]]])
        return (nxt[start:end] > c[start:end]).astype("float64")


class FirstCallCheater:
    """Cheats on the FIRST positions() call per (start,end) key (the SCORED call), honest on every
    later call (the gate's recomputes). Defeated only by gating the scored array (re-audit headline)."""
    name, horizon = "first_call_cheater", 1

    def __init__(self):
        self._seen: set = set()

    def fit(self, data, train_start, train_end):
        pass

    def positions(self, data, start, end):
        c = pd.to_numeric(data.candles["close"], errors="coerce").to_numpy("float64")
        key = (start, end)
        if key not in self._seen:
            self._seen.add(key)
            nxt = np.concatenate([c[1:], [c[-1]]])           # look-ahead on the SCORED call
            return (nxt[start:end] > c[start:end]).astype("float64")
        ema = pd.Series(c[:end]).ewm(span=24, adjust=False).mean().to_numpy()  # honest on gate recalls
        return (c[start:end] > ema[start:end]).astype("float64")


def test_fold_selective_cheater_now_disqualified():
    rep = run_benchmark(MiddleFoldOnlyHonest(), _data(8000), split_cfg=SPLIT_3F)
    assert rep["gates"]["scope"] == "all_folds" and rep["gates"]["n_folds_gated"] >= 3
    assert rep["disqualified"] is True
    mid = rep["gates"]["n_folds_gated"] // 2
    assert any(f != mid for f in rep["gates"]["failed_folds"])     # a non-middle fold caught it


def test_first_call_scored_array_cheater_disqualified():
    # the re-audit headline: scored on the cheat, gated on honesty. Caught only because the
    # determinism gate now compares the SCORED array against a recompute.
    rep = run_benchmark(FirstCallCheater(), _data(8000), split_cfg=SPLIT_3F)
    assert rep["gates"]["determinism"] is False
    assert rep["disqualified"] is True


def test_tiny_window_fold_fails_closed():
    from btc_benchmark.benchmark.validity import run_gates
    rep = run_gates(EmaRule(), _data(200), start=50, end=51, scored=np.array([0.0]))
    assert rep["future_perturbation"] is False and rep["prefix_invariance"] is False
    assert rep["passed"] is False


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


class TailGapCheater:
    """Honest on the first 75% of EVERY fold window; look-ahead in the tail (start+0.75n, end).
    The pre-fix gate only forward-checked cutoffs at 0.25/0.5/0.75n and NEVER the tail, so this
    scored a fraudulent return undisqualified (audit-3). Stride-1 coverage forward-checks every
    interior index, so the tail peek is caught."""
    name, horizon = "tail_gap_cheater", 1

    def fit(self, data, train_start, train_end):
        pass

    def positions(self, data, start, end):
        c = pd.to_numeric(data.candles["close"], errors="coerce").to_numpy("float64")
        n = end - start
        cut = start + int(n * 0.75)
        ema = pd.Series(c[:end]).ewm(span=24, adjust=False).mean().to_numpy()
        nxt = np.concatenate([c[1:], [c[-1]]])
        out = np.empty(n)
        for k, t in enumerate(range(start, end)):
            out[k] = (1.0 if c[t] > ema[t] else 0.0) if t <= cut \
                else (1.0 if nxt[t] > c[t] else 0.0)              # tail: peek tomorrow's close
        return out


def test_tail_gap_cheater_disqualified():
    # audit-3 regression: look-ahead confined to the tail (last 25%) of each fold. Fractional cutoffs
    # never forward-checked the tail; stride-1 coverage does.
    rep = run_benchmark(TailGapCheater(), _data(8000), split_cfg=SPLIT_3F)
    assert rep["gates"]["future_perturbation"] is False
    assert rep["disqualified"] is True


def test_single_interior_lookahead_caught_at_its_index():
    # the sharpest form: a position that peeks ONE bar ahead at a single interior index that lies
    # BETWEEN the old fractional cutoffs. Only stride-1 (a cutoff at every index) catches it.
    from btc_benchmark.benchmark.validity import run_gates
    data = _data(1000)
    c = pd.to_numeric(data.candles["close"], errors="coerce").to_numpy("float64")
    start, end, peek = 100, 400, 251                              # 251 avoids fracs 175/250/325

    class OneIndexPeek:
        name, horizon = "one_index_peek", 1

        def fit(self, *a):
            pass

        def positions(self, d, s, e):
            cc = pd.to_numeric(d.candles["close"], errors="coerce").to_numpy("float64")
            o = np.zeros(e - s)
            if s <= peek < e and peek + 1 < len(cc):             # robust to the prefix gate's short window
                o[peek - s] = 1.0 if cc[peek + 1] > cc[peek] else 0.0
            return o

    scored = OneIndexPeek().positions(data, start, end)
    rep = run_gates(OneIndexPeek(), data, start=start, end=end, scored=scored)
    assert rep["future_perturbation"] is False                   # caught by the cutoff at t0=251
    assert rep["passed"] is False


def test_non_ohlcv_future_column_cheater_disqualified():
    # the loader hands strategies the WHOLE processed frame (quote_volume, number_of_trades,
    # taker_buy_*, ...). A cheater can peek at the FUTURE row of any of those numeric columns, so the
    # perturber must perturb EVERY numeric column -- not just OHLCV. (audit self-review hole.)
    from btc_benchmark.benchmark.validity import run_gates
    data = _data(1000)
    rng = np.random.default_rng(7)
    cand = data.candles.copy()
    cand["number_of_trades"] = rng.uniform(1.0, 100.0, len(cand))   # causal benign non-OHLCV column
    data = BenchmarkData(candles=cand, aux=data.aux)
    start, end = 100, 400

    class FutureTradesCheater:
        name, horizon = "future_trades_cheater", 1

        def fit(self, *a):
            pass

        def positions(self, d, s, e):
            nt = pd.to_numeric(d.candles["number_of_trades"], errors="coerce").to_numpy("float64")
            o = np.zeros(e - s)
            for k, t in enumerate(range(s, e)):
                nxt = nt[t + 1] if t + 1 < len(nt) else nt[t]       # FUTURE row of a non-OHLCV column
                o[k] = 1.0 if nxt > nt[t] else 0.0
            return o

    scored = FutureTradesCheater().positions(data, start, end)
    rep = run_gates(FutureTradesCheater(), data, start=start, end=end, scored=scored)
    assert rep["future_perturbation"] is False                     # caught: the column's future is perturbed
    assert rep["passed"] is False


def test_cache_replay_lookahead_cheater_disqualified():
    # a STATEFUL cheater: computes look-ahead ONCE on the scored call and memoises it keyed by
    # (start, end), replaying that array on every later call. Fixed-window perturbation would hit the
    # cache and pass; the per-cutoff UNIQUE window [start, t0+2) forces a recompute on perturbed data.
    from btc_benchmark.benchmark.validity import run_gates
    data = _data(1000)
    start, end = 100, 400

    class CacheReplay:
        name, horizon = "cache_replay", 1

        def __init__(self):
            self._cache: dict = {}

        def fit(self, *a):
            pass

        def positions(self, d, s, e):
            key = (s, e)
            if key in self._cache:
                return self._cache[key]                     # replay -> ignores the perturbed data
            c = pd.to_numeric(d.candles["close"], errors="coerce").to_numpy("float64")
            nxt = np.concatenate([c[1:], [c[-1]]])
            out = (nxt[s:e] > c[s:e]).astype("float64")      # look-ahead, computed once on the clean call
            self._cache[key] = out
            return out

    strat = CacheReplay()
    scored = strat.positions(data, start, end)              # scored call caches the cheat under (100,400)
    rep = run_gates(strat, data, start=start, end=end, scored=scored)
    assert rep["future_perturbation"] is False              # unique short windows defeat the replay
    assert rep["passed"] is False


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
    # INVARIANT (flat/flip/scale): sum of per-trade cost_paid == portfolio cost, no double-count
    for pat in (np.array([0, 1, 1, 0, -1, -1, 0, 1, 0], "float64"),          # flat + flip
                np.array([0, 1, 0.5, 1, -0.3, -1, 0, 0.7, 0], "float64"),    # scale + fractional flip
                np.array([0, -1, -1, -1, 1, 1, 0, 0, 0], "float64")):        # direct flip mid-run
        m = len(pat)
        ts2 = pd.date_range("2022-01-01", periods=m, freq="1h", tz="UTC")
        cdl = pd.DataFrame({"timestamp_open": ts2,
                            "timestamp_close": ts2 + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
                            "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                            "volume": 1.0, "is_imputed": False})
        r = run_backtest(cdl, pat, CostConfig(fee_bps=10.0), periods_per_year=8760)
        assert abs(float(r.trades["cost_paid"].sum()) - float(r.costs.sum())) < 1e-12, pat.tolist()
