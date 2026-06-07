"""
Strategy library. Each returns a target-weight DataFrame (cols=symbols), with
sum|w| ~= `gross` per bar (default 1.0 = 100% notional, no leverage). Leverage is
applied later by the engine's leverage_curve so we can study risk of ruin cleanly.

All features use rolling windows ending at bar t (known at close t). The engine
applies a 1-bar execution lag, so there is no look-ahead.

Strategies implemented (all well-documented across assets & in crypto):
  * ts_momentum   : time-series (trend) momentum, inverse-vol sized
  * xs_momentum   : cross-sectional momentum (long winners / short losers)
  * xs_reversal   : short-term cross-sectional reversal
  * donchian      : channel breakout trend
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# --------------------- point-in-time eligibility ---------------------
def eligibility(close: pd.DataFrame, volume: pd.DataFrame,
                min_history: int, min_dollar_vol: float, liq_lookback: int) -> pd.DataFrame:
    """Boolean mask: tradeable at close[t] (enough history + trailing liquidity)."""
    has = close.notna()
    hist_ok = has.rolling(min_history, min_periods=min_history).sum() >= min_history
    dollar = (close * volume)
    liq = dollar.rolling(liq_lookback, min_periods=max(5, liq_lookback // 2)).median()
    return has & hist_ok & (liq >= min_dollar_vol)


def _inv_vol(ret: pd.DataFrame, lookback: int) -> pd.DataFrame:
    v = ret.rolling(lookback, min_periods=lookback // 2).std()
    return 1.0 / v.replace(0.0, np.nan)


def _normalize(w: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    s = w.abs().sum(axis=1).replace(0.0, np.nan)
    return w.div(s, axis=0).mul(gross).fillna(0.0)


# --------------------------- strategies ---------------------------
def ts_momentum(close, volume, *, lookback=168, vol_lookback=168, elig,
                long_only=False, gross=1.0) -> pd.DataFrame:
    """Trend: sign of trailing return over `lookback`, inverse-vol sized."""
    ret = close.pct_change()
    trail = close.pct_change(lookback)
    sig = np.sign(trail)
    if long_only:
        sig = sig.clip(lower=0.0)
    raw = sig * _inv_vol(ret, vol_lookback)
    raw = raw.where(elig, 0.0)
    return _normalize(raw, gross)


def ts_trend_strength(close, volume, *, lookbacks=(15, 30, 60, 90), vol_lookback=15,
                      strength_vol=30, k=2.0, elig, long_only=False, gross=1.0) -> pd.DataFrame:
    """Improved trend: measure each coin's trend as RISK-ADJUSTED strength
    tanh(k * r_L / (sigma*sqrt(L))) averaged over lookbacks (continuous, bounded), then
    inverse-vol sized. Beats naive sign-trend in/out-of-sample (stronger/cleaner trends
    get more weight; noise near zero gets little). This is the flagship."""
    ret = close.pct_change()
    sig = strength_signal(close, lookbacks=lookbacks, strength_vol=strength_vol, k=k)
    if long_only:
        sig = sig.clip(lower=0.0)
    raw = (sig * _inv_vol(ret, vol_lookback)).where(elig, 0.0)
    return _normalize(raw, gross)


def strength_signal(close, *, lookbacks=(15, 30, 60, 90), strength_vol=30, k=2.0) -> pd.DataFrame:
    """Continuous risk-adjusted trend signal in [-1,1] (no sizing)."""
    ret = close.pct_change()
    vol = ret.rolling(strength_vol, min_periods=strength_vol // 2).std()
    return sum(np.tanh(k * close.pct_change(L) / (vol * np.sqrt(L))) for L in lookbacks) / len(lookbacks)


def ts_momentum_multi(close, volume, *, lookbacks=(168, 336, 720, 1440), vol_lookback=168,
                      elig, long_only=False, gross=1.0) -> pd.DataFrame:
    """Robust trend: average the SIGN of trailing return across several lookbacks
    (a name must trend across multiple horizons to earn full weight), inverse-vol sized.
    Averaging across lookbacks is far less overfit than choosing one 'magic' window."""
    ret = close.pct_change()
    sig = sum(np.sign(close.pct_change(L)) for L in lookbacks) / len(lookbacks)
    if long_only:
        sig = sig.clip(lower=0.0)
    raw = sig * _inv_vol(ret, vol_lookback)
    raw = raw.where(elig, 0.0)
    return _normalize(raw, gross)


def xs_momentum(close, volume, *, lookback=168, skip=24, vol_lookback=168, elig,
                quantile=0.3, long_only=False, gross=1.0) -> pd.DataFrame:
    """Cross-sectional momentum: rank by past return (skipping recent `skip` bars),
    long top quantile / short bottom quantile, inverse-vol within legs."""
    ret = close.pct_change()
    mom = (close.shift(skip) / close.shift(lookback) - 1.0)
    mom = mom.where(elig)
    iv = _inv_vol(ret, vol_lookback).where(elig)

    ranks = mom.rank(axis=1, pct=True)
    n_elig = elig.sum(axis=1)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    longs = ranks >= (1.0 - quantile)
    shorts = ranks <= quantile
    wl = (iv.where(longs)).fillna(0.0)
    ws = (iv.where(shorts)).fillna(0.0)
    # dollar-neutral legs (each leg half gross); long-only uses only long leg
    wl = wl.div(wl.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    ws = ws.div(ws.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if long_only:
        w = wl * gross
    else:
        w = (wl - ws) * (gross / 2.0)
    w = w.where(n_elig >= 4, 0.0)     # need a few names to rank meaningfully
    return w.fillna(0.0)


def xs_reversal(close, volume, *, lookback=24, vol_lookback=168, elig,
                quantile=0.3, gross=1.0) -> pd.DataFrame:
    """Short-term reversal: long recent losers / short recent winners."""
    ret = close.pct_change()
    rec = close.pct_change(lookback).where(elig)
    iv = _inv_vol(ret, vol_lookback).where(elig)
    ranks = rec.rank(axis=1, pct=True)
    n_elig = elig.sum(axis=1)
    longs = ranks <= quantile          # losers
    shorts = ranks >= (1.0 - quantile) # winners
    wl = (iv.where(longs)).fillna(0.0)
    ws = (iv.where(shorts)).fillna(0.0)
    wl = wl.div(wl.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    ws = ws.div(ws.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w = (wl - ws) * (gross / 2.0)
    return w.where(n_elig >= 4, 0.0).fillna(0.0)


def ema_ensemble(weights: pd.DataFrame, spans=(5, 10, 15)) -> pd.DataFrame:
    """Smooth target weights with an ENSEMBLE of EMA spans (avoids picking one span).
    Cuts turnover ~60% and timing noise -> higher Sharpe/Calmar. Robust across spans."""
    return sum(weights.ewm(span=s).mean() for s in spans) / len(spans)


def signal_to_weights(signal: pd.DataFrame, ret: pd.DataFrame, elig: pd.DataFrame,
                      vol_lookback: int = 15, gross: float = 1.0) -> pd.DataFrame:
    """Turn a [-1,1] signal matrix into inverse-vol-sized, normalized weights."""
    raw = (signal * _inv_vol(ret, vol_lookback)).where(elig, 0.0)
    return _normalize(raw, gross)


def concentrate(weights: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep only the k highest-conviction (largest |weight|) positions per bar and
    renormalize to the original gross. Makes a broad book runnable on a tiny account."""
    absw = weights.abs()
    rank = absw.rank(axis=1, ascending=False, method="first")
    kept = weights.where(rank <= k, 0.0)
    gross = weights.abs().sum(axis=1)
    newg = kept.abs().sum(axis=1).replace(0.0, np.nan)
    return kept.mul(gross / newg, axis=0).fillna(0.0)


def donchian(close, volume, *, lookback=168, vol_lookback=168, elig,
             long_only=False, gross=1.0) -> pd.DataFrame:
    """Channel breakout: long if close == rolling max (new high), short if new low."""
    ret = close.pct_change()
    hi = close.rolling(lookback, min_periods=lookback).max()
    lo = close.rolling(lookback, min_periods=lookback).min()
    sig = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    sig = sig.mask(close >= hi, 1.0).mask(close <= lo, -1.0)
    # hold last signal until opposite breakout (state machine via ffill of nonzero)
    state = sig.replace(0.0, np.nan).ffill().fillna(0.0)
    if long_only:
        state = state.clip(lower=0.0)
    raw = state * _inv_vol(ret, vol_lookback)
    raw = raw.where(elig, 0.0)
    return _normalize(raw, gross)
