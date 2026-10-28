"""v4 strategy: 70% v3 risk-adjusted trend book + 30% BTC-hedged pump-follow sleeve.

A faithful port of research/final_v4.py (research/strategies.py, research/engine.py,
research/explore_v4.py, research/robust_v4.py). bot/tests/test_strategy_parity.py checks that
this module reproduces the research weights.

Input: aligned daily bars (index = UTC day, columns = symbols), CLOSED bars only.
Output: target weights for the last bar (fraction of equity per symbol, before leverage).
All features use data up to and including the last close: trade at/just after that close.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyCfg

COST_PER_SIDE = 0.0007            # 5bps taker + 2bps slippage (used only inside vol targeting)
FUNDING_ANNUAL = 0.11             # generic funding drag on net-long exposure
PUMP_FUNDING_STRESS = 0.50        # pumped coins: longs assumed to pay 50%/yr
TREND_LOOKBACKS = (15, 30, 60, 90)
TREND_K = 2.0
K_SELECT, K_HOLD, EMA_SPANS = 6, 10, (5, 10, 15)
BARS_PER_YEAR = 365.0


# ----------------------------------------------------------------------------- building blocks
def eligibility(close, volume, min_history, min_dollar_volume, liq_lookback=30):
    has = close.notna()
    hist_ok = has.rolling(min_history, min_periods=min_history).sum() >= min_history
    liq = (close * volume).rolling(liq_lookback, min_periods=max(5, liq_lookback // 2)).median()
    return has & hist_ok & (liq >= min_dollar_volume)


def _inv_vol(ret, lookback):
    v = ret.rolling(lookback, min_periods=lookback // 2).std()
    return 1.0 / v.replace(0.0, np.nan)


def _normalize(w, gross=1.0):
    s = w.abs().sum(axis=1).replace(0.0, np.nan)
    return w.div(s, axis=0).mul(gross).fillna(0.0)


def strength_signal(close, lookbacks=TREND_LOOKBACKS, strength_vol=30, k=TREND_K):
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(strength_vol, min_periods=strength_vol // 2).std()
    return sum(np.tanh(k * close.pct_change(L, fill_method=None) / (vol * np.sqrt(L)))
               for L in lookbacks) / len(lookbacks)


def concentrate(weights, k):
    rank = weights.abs().rank(axis=1, ascending=False, method="first")
    kept = weights.where(rank <= k, 0.0)
    gross = weights.abs().sum(axis=1)
    newg = kept.abs().sum(axis=1).replace(0.0, np.nan)
    return kept.mul(gross / newg, axis=0).fillna(0.0)


def ema_ensemble(weights, spans=EMA_SPANS):
    return sum(weights.ewm(span=s).mean() for s in spans) / len(spans)


def simulate_net(weights, ret, cost_per_side=COST_PER_SIDE, funding_matrix=None,
                 funding_annual=FUNDING_ANNUAL):
    """Per-bar net return of holding weights[t-1] over bar t (drift-aware costs, funding)."""
    W = weights.reindex_like(ret).fillna(0.0)
    R = ret.fillna(0.0)
    Wp = W.shift(1).fillna(0.0)
    gross = (Wp * R).sum(axis=1)
    drifted = (Wp * (1.0 + R)).div(1.0 + gross, axis=0).fillna(0.0)
    cost = ((W - drifted).abs().sum(axis=1) * cost_per_side).shift(1).fillna(0.0)
    if funding_matrix is not None:
        funding = (Wp * funding_matrix.reindex_like(ret).fillna(0.0)).sum(axis=1)
    else:
        funding = Wp.sum(axis=1) * (funding_annual / BARS_PER_YEAR)
    return gross - cost - funding.fillna(0.0)


def vol_target(weights, ret, target_vol, max_leverage, funding_matrix=None, lookback=30):
    """Scale weights so the strategy's own trailing realized vol ~= target_vol (lagged 1 bar)."""
    base = simulate_net(weights, ret, funding_matrix=funding_matrix)
    rv = base.rolling(lookback, min_periods=lookback // 2).std() * np.sqrt(BARS_PER_YEAR)
    scale = (target_vol / rv).shift(1).clip(upper=max_leverage).fillna(0.0)
    return weights.mul(scale, axis=0)


# ----------------------------------------------------------------------------- sleeves
def trend_weights(close, elig):
    ret = close.pct_change(fill_method=None)
    sig = strength_signal(close)
    raw = (sig * _inv_vol(ret, 15)).where(elig, 0.0)
    w6 = concentrate(_normalize(raw), K_SELECT)
    return concentrate(ema_ensemble(w6), K_HOLD)


def pump_weights(close, elig, hedge, threshold, hold):
    """Long coins that beat the hedge (BTC) by > threshold in one day, for `hold` days,
    inverse-vol sized, with an equal-notional short in the hedge. Returns (weights, funding)."""
    ret = close.pct_change(fill_method=None)
    idio = ret.sub(ret[hedge], axis=0)
    trig = (idio > threshold) & elig
    held = trig.astype(float).rolling(hold, min_periods=1).max().astype(bool) & elig
    vol = ret.rolling(20, min_periods=5).std()
    raw = (1.0 / vol).where(held, 0.0).fillna(0.0)
    w = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w[hedge] = w[hedge] - w.sum(axis=1)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    generic = FUNDING_ANNUAL / BARS_PER_YEAR
    funding = pd.DataFrame(-generic, index=close.index, columns=close.columns)
    funding = funding.mask(held, PUMP_FUNDING_STRESS / BARS_PER_YEAR)
    return w, funding


# ----------------------------------------------------------------------------- entry point
@dataclass
class StrategyOutput:
    as_of: pd.Timestamp
    weights: pd.Series            # final blended target weights (sum|w| = gross, pre-leverage)
    trend: pd.Series
    pump: pd.Series
    n_eligible: int
    pump_names: list


def compute_weight_history(close: pd.DataFrame, volume: pd.DataFrame, cfg: StrategyCfg):
    """Full history of target weights (DataFrames). Used by compute_targets and by tests."""
    close = close.astype("float64")
    volume = volume.reindex_like(close).astype("float64")
    drop = [c for c in close.columns if _base(c) in set(cfg.exclude)]
    close, volume = close.drop(columns=drop), volume.drop(columns=drop)
    if cfg.hedge_symbol not in close.columns:
        raise ValueError(f"hedge symbol {cfg.hedge_symbol} missing from data")
    ret = close.pct_change(fill_method=None)
    elig = eligibility(close, volume, cfg.min_history, cfg.min_dollar_volume)
    wt = vol_target(trend_weights(close, elig), ret, cfg.target_vol, cfg.max_sleeve_leverage)
    pw, pf = pump_weights(close, elig, cfg.hedge_symbol, cfg.pump_threshold, cfg.pump_hold_days)
    wp = vol_target(pw, ret, cfg.target_vol, cfg.max_sleeve_leverage, funding_matrix=pf)
    blend = cfg.w_trend * wt + cfg.w_pump * wp
    return blend, wt, wp, elig, pw


def compute_targets(close: pd.DataFrame, volume: pd.DataFrame, cfg: StrategyCfg) -> StrategyOutput:
    blend, wt, wp, elig, pw = compute_weight_history(close, volume, cfg)
    t = blend.index[-1]
    nz = lambda s: s[s.abs() > 1e-9]
    held = pw.iloc[-1]
    return StrategyOutput(
        as_of=t, weights=nz(blend.iloc[-1]), trend=nz(wt.iloc[-1]), pump=nz(wp.iloc[-1]),
        n_eligible=int(elig.iloc[-1].sum()),
        pump_names=sorted(held[held > 0].index.tolist()))


def _base(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol
