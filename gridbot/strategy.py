"""Decides each new grid's spacing and bias, and whether a grid may run at all.

Same formulas as the research (research/grid_study.py):
  sigma_1h        : std of hourly log returns over the last 168 closed 1h bars
  trend strength  : v3 risk-adjusted daily trend, mean of tanh(2 * r_L / (sigma_30d * sqrt(L)))
                    over L = 15, 30, 60, 90 days, in [-1, 1]
"""
from __future__ import annotations
import logging
import math
import time

import numpy as np

from .config import GridCfg
from .engine import GridPlan

log = logging.getLogger(__name__)
LOOKBACKS = (15, 30, 60, 90)


def hourly_sigma(closes: list[float]) -> float:
    c = np.asarray(closes[-169:], dtype=float)
    r = np.diff(np.log(c))
    r = r[np.isfinite(r)]
    return float(np.std(r, ddof=1)) if len(r) >= 48 else float("nan")


def trend_strength(daily_closes: list[float], k: float = 2.0) -> float:
    c = np.asarray(daily_closes, dtype=float)
    if len(c) < max(LOOKBACKS) + 31:
        return float("nan")
    r = np.diff(c) / c[:-1]
    vol = np.std(r[-30:], ddof=1)
    if not vol > 0:
        return float("nan")
    return float(np.mean([math.tanh(k * (c[-1] / c[-1 - L] - 1) / (vol * math.sqrt(L))) for L in LOOKBACKS]))


class GridStrategy:
    """`klines(symbol, interval, limit)` must return Binance kline rows (closed bars only)."""

    def __init__(self, cfg: GridCfg, klines, cache_seconds: int = 300, clock=time.time):
        self.cfg, self.klines, self.cache_seconds, self.clock = cfg, klines, cache_seconds, clock
        self._cache: dict = {}

    def _closes(self, symbol, interval, limit):
        key = (symbol, interval)
        hit = self._cache.get(key)
        if hit and self.clock() - hit[0] < self.cache_seconds:
            return hit[1]
        rows = self.klines(symbol, interval, limit)
        closes = [float(k[4]) for k in rows]
        self._cache[key] = (self.clock(), closes)
        return closes

    def trend(self, symbol) -> float:
        return trend_strength(self._closes(symbol, "1d", 130))

    def spacing(self, symbol) -> float:
        c = self.cfg
        if c.spacing == "fixed":
            return c.spacing_fixed
        s = hourly_sigma(self._closes(symbol, "1h", 200))
        if not math.isfinite(s):
            return float("nan")
        return min(max(c.spacing_k * s, c.spacing_min), c.spacing_max)

    def regime_ok(self, symbol) -> bool:
        if not self.cfg.regime_filter:
            return True
        t = self.trend(symbol)
        return math.isfinite(t) and abs(t) < self.cfg.trend_threshold

    def plan(self, symbol) -> GridPlan:
        c = self.cfg
        g = self.spacing(symbol)
        if not math.isfinite(g):
            return GridPlan(False, reason="not enough 1h history for volatility")
        bias = c.bias
        t = None
        if c.bias == "trend" or c.regime_filter:
            t = self.trend(symbol)
            if not math.isfinite(t):
                return GridPlan(False, reason="not enough daily history for the trend")
        if c.regime_filter and abs(t) >= c.trend_threshold:
            return GridPlan(False, reason=f"trending (strength {t:+.2f})")
        if c.bias == "trend":
            bias = "long" if t > c.trend_threshold else ("short" if t < -c.trend_threshold else "neutral")
        return GridPlan(True, g=g, bias=bias, reason=f"g={g:.4f} bias={bias}" + (f" trend={t:+.2f}" if t is not None else ""))
