"""Milestone 5.5 tests: PSR / Deflated Sharpe / PBO sanity."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_benchmark.backtest.overfitting import (  # noqa: E402
    block_bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    minimum_track_record_length,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    sharpe_per_period,
)


def test_block_bootstrap_sharpe_ci_separates_signal_from_noise():
    rng = np.random.default_rng(0)
    good = rng.normal(0.0008, 0.01, 20000)   # positive-Sharpe series
    noise = rng.normal(0.0, 0.01, 20000)
    cg = block_bootstrap_sharpe_ci(good, block=168, n_boot=500, periods_per_year=8760)
    cn = block_bootstrap_sharpe_ci(noise, block=168, n_boot=500, periods_per_year=8760)
    assert cg["sharpe_lo"] > 0 and cg["p_positive"] > 0.95     # robustly positive
    assert cn["sharpe_lo"] < 0 < cn["sharpe_hi"]               # noise CI straddles 0


def test_psr_separates_good_from_bad():
    rng = np.random.default_rng(0)
    good = rng.normal(0.001, 0.01, 5000)
    bad = rng.normal(-0.001, 0.01, 5000)
    assert probabilistic_sharpe_ratio(good, 0.0) > 0.9
    assert probabilistic_sharpe_ratio(bad, 0.0) < 0.1


def test_dsr_penalizes_more_trials():
    rng = np.random.default_rng(1)
    ret = rng.normal(0.0006, 0.01, 6000)
    sr = sharpe_per_period(ret)
    sr_few = np.array([sr * 0.6, sr, sr * 1.1])
    sr_many = np.concatenate([[sr, sr], rng.normal(sr, abs(sr) + 0.01, 300)])
    d_few = deflated_sharpe_ratio(ret, sr_few, n_trials=3)["dsr"]
    d_many = deflated_sharpe_ratio(ret, sr_many, n_trials=300)["dsr"]
    assert d_many <= d_few  # more trials => stricter (lower DSR)


def test_pbo_robust_vs_overfit():
    rng = np.random.default_rng(2)
    S, K = 10, 20
    M_robust = rng.normal(0, 1, (S, K)); M_robust[:, 0] += 2.5   # config 0 genuinely best everywhere
    M_noise = rng.normal(0, 1, (S, K))                          # pure noise => overfit selection
    assert pbo_cscv(M_robust)["pbo"] < 0.2
    assert pbo_cscv(M_noise)["pbo"] > 0.3


def test_mintrl_inverts_psr():
    # at exactly T = MinTRL, the PSR(benchmark) must equal `confidence` -> the formula is a true inverse
    from scipy.stats import kurtosis, norm, skew
    rng = np.random.default_rng(3)
    r = rng.normal(0.0006, 0.01, 8000)
    out = minimum_track_record_length(r, sr_benchmark=0.0, confidence=0.95)
    assert np.isfinite(out["min_trl_periods"]) and out["min_trl_periods"] > 0
    sr = sharpe_per_period(r); g3 = float(skew(r)); g4 = float(kurtosis(r, fisher=False))
    n = out["min_trl_periods"]
    z = sr * np.sqrt(n - 1) / np.sqrt(1 - g3 * sr + ((g4 - 1) / 4) * sr * sr)
    assert abs(float(norm.cdf(z)) - 0.95) < 1e-6


def test_mintrl_sufficiency_and_benchmark():
    rng = np.random.default_rng(4)
    strong = rng.normal(0.004, 0.01, 5000)    # per-period Sharpe ~0.4 -> tiny MinTRL -> sufficient
    weak = rng.normal(0.0001, 0.01, 2000)     # ~zero Sharpe -> MinTRL huge/inf -> insufficient
    assert minimum_track_record_length(strong, 0.0, 0.95)["sufficient"]
    assert not minimum_track_record_length(weak, 0.0, 0.95)["sufficient"]
    # a Sharpe at/below the benchmark can never reach confidence above it -> infinite MinTRL
    neg = minimum_track_record_length(rng.normal(-0.002, 0.01, 3000), 0.0, 0.95)
    assert neg["min_trl_periods"] == float("inf") and not neg["sufficient"]
