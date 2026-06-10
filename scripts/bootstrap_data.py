"""One-command data bootstrap for benchmark users (clone -> this -> run strategies).

Rebuilds the full data bundle from Binance Data Vision (public, checksum-verified):
  1h klines -> validate -> impute -> processed canonical   (~5 MB, required)
  funding / open-interest / mark-premium                    (aux, recommended)
  5m / 15m / 1m sub-klines                                  (aux, optional: --with-sub-bars)

    python -m scripts.bootstrap_data                  # required + derivatives (~10 min)
    python -m scripts.bootstrap_data --with-sub-bars  # + 1m/5m/15m (~+15 min, ~250 MB)

The benchmark runs without sub-bars (load_benchmark_data(include_sub_bars=False)); strategies that
use intra-hour features need them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

START, SYMBOL, MARKET = "2019-09-01", "BTCUSDT", "futures_um"


def run(mod: str, *args: str) -> None:
    cmd = [sys.executable, "-m", mod, *args]
    print(f"\n=== {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-sub-bars", action="store_true", help="also download 1m/5m/15m klines")
    ap.add_argument("--start", default=START)
    args = ap.parse_args()
    common = ["--market", MARKET, "--symbol", SYMBOL]

    run("btc_benchmark.data.download_binance", *common, "--interval", "1h", "--start", args.start)
    run("btc_benchmark.data.validate_data", *common, "--interval", "1h")
    run("btc_benchmark.data.impute", *common, "--interval", "1h", "--policy", "flat_bar_fill")
    # per-source Data Vision history starts (earlier months just don't exist upstream)
    for kind, kstart in (("funding", "2021-11-01"), ("open_interest", "2021-01-01"),
                         ("mark_premium", "2020-01-01")):
        run("btc_benchmark.data.download_derivatives", "--kind", kind, "--symbol", SYMBOL, "--start", kstart)
    if args.with_sub_bars:
        for itv in ("15m", "5m", "1m"):
            run("btc_benchmark.data.download_binance", *common, "--interval", itv, "--start", "2020-01-01")
    print("\nbootstrap complete. Quick check: python -c \"from btc_benchmark.benchmark import "
          "load_benchmark_data; d=load_benchmark_data('.'); print(len(d.candles), list(d.aux))\"")


if __name__ == "__main__":
    main()
