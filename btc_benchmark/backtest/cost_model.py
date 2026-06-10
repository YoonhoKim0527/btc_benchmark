"""Transaction cost model.

Cost is proportional to turnover. Turnover into position p_t is |p_t - p_{t-1}| (p_{-1}=0):
  long->cash = 1, cash->long = 1, long->short = 2, short->long = 2, no-change = 0.
cost_t = turnover_t * all_in_cost_bps / 10000, where
  all_in_cost_bps = (fee_bps + slippage_bps + safety_buffer_bps) * cost_multiplier.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CostConfig:
    fee_bps: float = 10.0
    slippage_bps: float = 0.0
    safety_buffer_bps: float = 0.0
    funding_enabled: bool = False
    cost_multiplier: float = 1.0
    funding_sign: str = field(default="long_pays_when_positive")  # documented sign convention

    @property
    def all_in_cost_bps(self) -> float:
        return (self.fee_bps + self.slippage_bps + self.safety_buffer_bps) * self.cost_multiplier

    def scaled(self, multiplier: float) -> "CostConfig":
        """Return a copy with cost_multiplier set (for 0x/0.5x/1x/2x/3x sensitivity)."""
        return CostConfig(self.fee_bps, self.slippage_bps, self.safety_buffer_bps,
                          self.funding_enabled, multiplier, self.funding_sign)


def turnover_series(positions) -> np.ndarray:
    """|p_t - p_{t-1}| with p_{-1}=0 (entering the first position counts as turnover)."""
    p = np.asarray(positions, dtype="float64")
    prev = np.concatenate([[0.0], p[:-1]])
    return np.abs(p - prev)


def cost_series(positions, cfg: CostConfig) -> np.ndarray:
    return turnover_series(positions) * cfg.all_in_cost_bps / 10000.0
