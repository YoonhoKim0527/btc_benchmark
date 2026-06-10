# btc_benchmark

BTC 1h 트레이딩 벤치마크의 **동결된 심판(referee)**. 참가자는
[`btc_agentic_system`](https://github.com/YoonhoKim0527/btc_agentic_system)을 포크해 전략만
업그레이드하고, 이 레포가 평가합니다. **이 레포의 코드/비용/분할/게이트는 버전으로 동결됩니다**
(`BENCHMARK_VERSION`, 점수에 항상 기록).

> Prime directive: **correct data + correct backtesting > high reported returns.**
> 누출·과적합·숨은 비용 인하·운 좋은 시드에서 나온 수익은 무효입니다.

## 무엇이 들어있나
- `btc_benchmark/data` — Binance Data Vision 다운로드(체크섬 검증) → validate → impute →
  canonical 1h + funding/OI/premium + 1m/5m/15m
- `btc_benchmark/backtest` — 감사된 백테스터(비용·펀딩·next-open), walk-forward(purge/embargo +
  **sealed holdout**), 메트릭, 과적합 진단(PSR/DSR/PBO/MinTRL/CPCV/블록부트스트랩)
- `btc_benchmark/benchmark` — **Strategy 계약** + **인과성 게이트**(결정성 / future-perturbation /
  prefix-invariance; 실패 = disqualified) + 채점 러너(비용 0–5x, next-open, funding-aware,
  random-matched percentile, 연도별) + 리더보드 JSONL
- `btc_benchmark/decisions` — 룰 베이스라인(buy-hold/EMA/Donchian/RSI) + 턴오버-매칭 랜덤 베이스라인

## Setup (주최자/참가자 공통)

```bash
git clone https://github.com/YoonhoKim0527/btc_benchmark.git
cd btc_benchmark
pip install -e ".[dev]"
pytest                                   # 심판을 신뢰하기 전에 테스트부터
python -m scripts.bootstrap_data         # 데이터 번들 재구축 (1h+파생 ~10분)
python -m scripts.bootstrap_data --with-sub-bars   # 선택: 1m/5m/15m (intra-hour 전략용)
```

## 전략 평가 (참가자는 btc_agentic_system에서)

```python
from btc_benchmark import load_benchmark_data, run_benchmark
data = load_benchmark_data("/path/to/btc_benchmark")     # 데이터 번들이 있는 체크아웃
report = run_benchmark(MyStrategy(), data, team="myteam",
                       leaderboard_path="results/leaderboard.jsonl")
```

규칙 (코드로 강제):
1. 벤치마크가 walk-forward를 직접 구동 — 전략은 분할을 선택할 수 없고 fold 밖을 보지 못함
2. 인과성 게이트 통과 전에는 점수 없음 (실패 = `disqualified`, 기록은 남음)
3. 비용·회계·분할은 심판 상수 — 제출물이 변경 불가
4. **Sealed holdout(마지막 6개월)은 dev 평가에 구조적으로 비노출** — one-shot 최종평가는 주최자만

## 계보
`btc_autoresearch`(M1–M7 연구 모노레포)에서 심판 서브셋을 카브했습니다. 전체 연구 히스토리/감사
기록은 그쪽에 있습니다.
