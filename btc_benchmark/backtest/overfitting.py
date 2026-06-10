"""Backtest-overfitting diagnostics (López de Prado): PSR, Deflated Sharpe, PBO via CSCV.

All formulas are documented; approximations are flagged. These DISCOUNT a result for the number
of trials tried and for non-normality / sample length -- they do not "prove" a strategy works.
Sharpe inputs are PER-PERIOD (not annualized); use perperiod = annualized / sqrt(periods_per_year).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import kurtosis, norm, skew

_GAMMA = 0.5772156649015329  # Euler-Mascheroni


def sharpe_per_period(returns) -> float:
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """PSR(SR0): P(true per-period SR > SR0), correcting for skew/kurtosis and sample length.

    PSR = Phi[ (SR - SR0)*sqrt(T-1) / sqrt(1 - g3*SR + ((g4-1)/4)*SR^2) ]
    g3 = skew, g4 = (non-excess) kurtosis, T = #returns.
    """
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 3:
        return float("nan")
    sr = sharpe_per_period(r)
    if not np.isfinite(sr):
        return float("nan")
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))  # non-excess (normal -> 3)
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    if denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(T - 1) / np.sqrt(denom)
    return float(norm.cdf(z))


def minimum_track_record_length(returns, sr_benchmark: float = 0.0, confidence: float = 0.95) -> dict:
    """MinTRL (López de Prado 2012): #observations needed for PSR(SR_benchmark) >= `confidence`.

    Inverts the PSR formula for T:
      MinTRL = 1 + [1 - g3*SR + ((g4-1)/4)*SR^2] * (Z_confidence / (SR - SR_benchmark))^2
    SR, SR_benchmark are PER-PERIOD; g3=skew, g4=non-excess kurtosis. Tells you whether a sample is long
    enough to call a Sharpe significant: if observed T < MinTRL, the track record is too short to confirm
    the edge at `confidence` even if real (a statistical-power check, complementary to the DSR's
    multiple-testing check). Returns nan MinTRL if SR <= benchmark (cannot reach confidence above it).
    """
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 3:
        return {"min_trl_periods": float("nan"), "observed_periods": T, "sufficient": False}
    sr = sharpe_per_period(r)
    if not np.isfinite(sr) or sr <= sr_benchmark:
        return {"min_trl_periods": float("inf"), "observed_periods": T, "sufficient": False,
                "sharpe_per_period": float(sr) if np.isfinite(sr) else float("nan")}
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))
    z = float(norm.ppf(confidence))
    num = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    if num <= 0:
        return {"min_trl_periods": float("nan"), "observed_periods": T, "sufficient": False}
    min_trl = 1.0 + num * (z / (sr - sr_benchmark)) ** 2
    return {"min_trl_periods": float(min_trl), "observed_periods": int(T),
            "sufficient": bool(T >= min_trl), "sharpe_per_period": float(sr), "confidence": confidence}


def expected_max_sharpe(trial_sharpes_per_period, n_trials: int | None = None) -> float:
    """E[max SR] under the null over N trials ~ sqrt(Var(SR_n)) * [(1-g)Z^-1(1-1/N) + g Z^-1(1-1/(N e))]."""
    s = np.asarray(trial_sharpes_per_period, dtype="float64")
    s = s[np.isfinite(s)]
    N = int(n_trials) if n_trials else len(s)
    if len(s) < 2 or N < 2:
        return 0.0
    var_sr = float(np.var(s, ddof=1))
    return float(np.sqrt(var_sr) * ((1 - _GAMMA) * norm.ppf(1 - 1.0 / N)
                                    + _GAMMA * norm.ppf(1 - 1.0 / (N * np.e))))


def deflated_sharpe_ratio(returns, trial_sharpes_per_period, n_trials: int | None = None) -> dict:
    """DSR = PSR(E[max SR]) -- probability the candidate's SR beats the best-of-N-trials null.

    Returns dict(dsr, expected_max_sr, candidate_sr_per_period, n_trials). DSR>0.95 => significant
    after accounting for the trial count.
    """
    e_max = expected_max_sharpe(trial_sharpes_per_period, n_trials)
    return {
        "dsr": probabilistic_sharpe_ratio(returns, sr_benchmark=e_max),
        "expected_max_sr_per_period": e_max,
        "candidate_sr_per_period": sharpe_per_period(returns),
        "n_trials": int(n_trials) if n_trials else int(np.isfinite(np.asarray(trial_sharpes_per_period)).sum()),
    }


def pbo_cscv(perf_matrix: np.ndarray, max_combos: int = 5000) -> dict:
    """Probability of Backtest Overfitting via Combinatorial Symmetric Cross-Validation.

    perf_matrix: shape (S_blocks, K_configs) of per-block performance (e.g. per-block mean return).
    Split the S blocks into all symmetric IS/OOS halves; for each, the IS-best config's OOS relative
    rank gives a logit; PBO = fraction of splits where the IS-best lands below the OOS median.
    PBO ~0 robust, ~0.5 pure overfitting. (Approximation: blocks, not full purged CV.)
    """
    M = np.asarray(perf_matrix, dtype="float64")
    S, K = M.shape
    if S < 2 or K < 2:
        return {"pbo": float("nan"), "n_splits": 0, "note": "need >=2 blocks and >=2 configs"}
    half = S // 2
    combos = list(combinations(range(S), half))
    truncated = False
    if len(combos) > max_combos:
        combos = combos[:max_combos]
        truncated = True
    logits = []
    for IS in combos:
        ISset = set(IS)
        OOS = [b for b in range(S) if b not in ISset]
        is_perf = M[list(IS)].mean(axis=0)
        oos_perf = M[OOS].mean(axis=0)
        n_star = int(np.argmax(is_perf))
        # strict mid-rank relative position of the IS-best in OOS (textbook PBO convention)
        rank = float((np.sum(oos_perf < oos_perf[n_star]) + 0.5) / len(oos_perf))
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.asarray(logits)
    return {"pbo": float(np.mean(logits <= 0)), "n_splits": len(combos),
            "median_logit": float(np.median(logits)), "blocks": S, "configs": K,
            "truncated": truncated}


def block_bootstrap_sharpe_ci(returns, *, block: int = 168, n_boot: int = 2000, seed: int = 0,
                              ci: float = 0.95, periods_per_year: int = 8760) -> dict:
    """Circular block-bootstrap CI for the (annualized) Sharpe -- non-parametric, autocorrelation-aware.

    Resamples contiguous blocks (default 168h = 1 week) with wrap-around, recomputing the Sharpe each
    time. A CI lower bound > 0 means the Sharpe is robustly positive accounting for serial dependence.
    This does NOT correct for multiple testing (that is the Deflated Sharpe's job) -- it is a
    single-series uncertainty band.
    """
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    n = len(r)
    if n < block * 2:
        return {"sharpe_lo": float("nan"), "sharpe_hi": float("nan"), "p_positive": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offs = np.arange(block)
    sh = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offs[None, :]).ravel() % n)[:n]
        s = r[idx]
        sd = s.std(ddof=1)
        if sd > 0:
            sh.append(s.mean() / sd * np.sqrt(periods_per_year))
    sh = np.asarray(sh)
    lo, hi = np.percentile(sh, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"sharpe_lo": float(lo), "sharpe_hi": float(hi), "sharpe_median": float(np.median(sh)),
            "p_positive": float(np.mean(sh > 0)), "block": block, "n_boot": int(len(sh)), "n": n}
