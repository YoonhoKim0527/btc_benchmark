# btc_benchmark

A **frozen referee** for the BTC 1h trading benchmark. Participants fork & upgrade only the
[`btc_agentic_system`](https://github.com/YoonhoKim0527/btc_agentic_system) repo; **this** repo
evaluates submissions. The data, backtester, costs, splits, causality gates, and sealed-holdout
protocol here are **version-frozen** (`BENCHMARK_VERSION`, stamped into every score).

> **Prime directive: correct data + correct backtesting > high reported returns.**
> Returns produced by lookahead, overfitting, a hidden cost reduction, or a lucky seed are invalid.

*(한글) BTC 1시간봉 트레이딩 벤치마크의 **동결된 심판**. 참가자는
`btc_agentic_system`만 포크해서 전략을 업그레이드하고, 이 레포가 평가합니다. 데이터·백테스터·
비용·분할·인과성 게이트·sealed holdout은 버전으로 동결됩니다.*

## What's inside
- `btc_benchmark/data` — Binance Data Vision downloader (streaming SHA-256 vs `.CHECKSUM`, atomic
  writes) → validate → impute (explicit flags, nothing silent) → canonical 1h, plus funding /
  open-interest / mark-premium and optional 1m/5m/15m sub-klines (backward as-of, staleness→NaN+flag).
- `btc_benchmark/backtest` — audited vectorized backtester (close-to-close *and* next-open execution;
  event-based 8h funding; per-trade log), cost model, metrics, walk-forward (purge + embargo +
  **sealed holdout**), overfitting diagnostics (PSR / DSR / PBO / MinTRL / CPCV / block-bootstrap CI).
- `btc_benchmark/benchmark` — the **Strategy contract**, the **causality gates** (run on **every**
  fold; failure ⇒ disqualified), and the scoring runner (official 1x metrics + cost 0–5×, next-open,
  funding-aware, turnover-matched random percentile, per-year) → leaderboard JSONL with the version.
- `btc_benchmark/decisions` — rule baselines (buy-hold / EMA / Donchian / RSI) + turnover-matched
  random baseline.

## Setup (host & participants)

```bash
git clone https://github.com/YoonhoKim0527/btc_benchmark.git
cd btc_benchmark
pip install -e ".[dev]"
pytest                                    # trust the referee only after its tests pass
python -m scripts.bootstrap_data          # rebuild the data bundle (1h + funding/OI/premium, ~10 min)
python -m scripts.bootstrap_data --with-sub-bars   # optional: + 1m/5m/15m for intra-hour strategies
```

`bootstrap_data` downloads raw → imputes (raw→processed) → validates (processed), with `--gapfill`
for months absent from the Data Vision monthly archives (e.g. 2019). Funding history on Data Vision
effectively begins 2021-11 (OI 2021-01, premium 2020-01); the bundle uses each source's earliest
available month, so funding-aware figures cover 2021-11 onward.

## Evaluating a strategy

```python
from btc_benchmark import load_benchmark_data, run_benchmark
data = load_benchmark_data("/path/to/btc_benchmark")     # a checkout whose data bundle is built
report = run_benchmark(MyStrategy(), data, team="myteam",
                       leaderboard_path="results/leaderboard.jsonl")
```

Rules (enforced in code, not by trust):
1. The benchmark drives the walk-forward; a strategy cannot pick its own splits or see beyond the
   fold it is asked about.
2. **Causality gates run on every fold** (determinism / future-perturbation incl. straddling sub-bars
   / prefix-invariance). Any failure ⇒ `disqualified` (the run is still recorded).
3. Costs, accounting, and splits are referee constants — a submission cannot change them.
4. The sealed holdout (last 6 months) is **structurally invisible** to dev evaluation; the one-shot
   final test is run only by the host.

### Trusted-but-unenforced boundaries (disclosed)
- **Declared horizon**: a strategy declares its label horizon (used for purge/embargo); deliberately
  under-declaring is the one channel the gates cannot see — it is recorded in the report and auditable.

## Lineage
Carved from `btc_autoresearch` (the M1–M7 research monorepo). Full research history and audit trail
live there; this repo is only the referee.
