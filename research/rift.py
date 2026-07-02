"""RIFT sleeve — cross-sectional momentum long/short (validated on branch
claude/unconventional-trading-strategy-dxl3r5, see seismo/README.md).

Long the n_side strongest 30d performers among eligible perps, short the
n_side weakest; inverse-vol weights with a per-name cap inside each leg;
dollar-neutral; targets refresh on a fixed 7-day calendar anchor (weekly
cadence is part of the validated spec -- daily refresh loses ~0.3 Sharpe
to turnover) and are held (ffill) in between.

Shared by the live bot (bot/signals.py) and the offline backtest
(seismo/combined.py) so both trade the identical construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _leg_weights(vols, budget, w_cap=0.20):
    """Inverse-vol weights for one leg with an iterative per-name cap."""
    iv = 1.0 / np.clip(vols, 1e-4, None)
    w = budget * iv / iv.sum()
    cap = w_cap * budget
    for _ in range(4):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over
        if not free.any() or w[free].sum() <= 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    return w


def rift_weights(close: pd.DataFrame, ret: pd.DataFrame, elig: pd.DataFrame, *,
                 lookback: int = 30, n_side: int = 10, w_cap: float = 0.20,
                 vol_lookback: int = 30, cadence_days: int = 7,
                 gross: float = 1.0) -> pd.DataFrame:
    """Target-weight matrix (date x symbol), sum|w| = gross on rebal days."""
    mom = np.log(close.astype("float64")).diff().rolling(lookback).sum().where(elig)
    rvol = ret.rolling(vol_lookback, min_periods=vol_lookback // 2).std()

    n_bars, n_sym = close.shape
    momv, volv, eligv = mom.values, rvol.values, elig.values
    epoch_days = np.asarray(close.index.normalize().asi8) // 86_400_000_000_000

    out = np.full((n_bars, n_sym), np.nan)
    for i in range(n_bars):
        if epoch_days[i] % cadence_days != 0:
            continue
        row = np.where(eligv[i] & np.isfinite(momv[i]) & np.isfinite(volv[i]),
                       momv[i], np.nan)
        ok = np.flatnonzero(np.isfinite(row))
        w = np.zeros(n_sym)
        if len(ok) >= 2 * n_side + 5:
            order = ok[np.argsort(row[ok])]
            shorts, longs = order[:n_side], order[-n_side:]
            w[longs] = _leg_weights(volv[i][longs], gross / 2, w_cap)
            w[shorts] = -_leg_weights(volv[i][shorts], gross / 2, w_cap)
        out[i] = w
    W = pd.DataFrame(out, index=close.index, columns=close.columns)
    return W.ffill().fillna(0.0)
