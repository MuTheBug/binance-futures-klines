"""Shared machinery for the v5 study (see V5_PROTOCOL.md). Same rules as engine.py / final_v4.py,
generalised to per-coin costs and real funding, on the blind vision panel."""
from __future__ import annotations
import numpy as np
import pandas as pd

import strategies as st
import vision_panel as vp
from final_v3 import K_HOLD, K_SELECT, KSTEEP, LBS, SPANS

BPY = 365.0
COST = 0.0007
START = pd.Timestamp("2020-06-01", tz="UTC")
IS_END, DESIGN_END = vp.IS_END, vp.DESIGN_END
# Frozen on 2026-11-10 after the classifier calibration (v5_classifier.py; V5_PROTOCOL.md,
# Deviations), before any strategy result was computed.
WEEKEND_THR = 0.35


def simulate(W, R, cost=COST, F=None, funding_annual=0.11) -> pd.Series:
    """Net daily return of holding W[t-1] over day t. cost: scalar or per-coin frame (per side).
    F: per-coin funding of each holding day (longs pay when positive); None = generic drag."""
    W = W.reindex_like(R).fillna(0.0)
    Rz = R.fillna(0.0)
    Wp = W.shift(1).fillna(0.0)
    gross = (Wp * Rz).sum(axis=1)
    drifted = (Wp * (1.0 + Rz)).div(1.0 + gross, axis=0).fillna(0.0)
    turn = (W - drifted).abs()
    c = (turn * cost.reindex_like(R)).sum(axis=1) if isinstance(cost, pd.DataFrame) else turn.sum(axis=1) * cost
    c = c.shift(1).fillna(0.0)
    if F is not None:
        fund = (Wp * F.reindex_like(R).fillna(0.0)).sum(axis=1)
    else:
        fund = Wp.sum(axis=1) * (funding_annual / BPY)
    return gross - c - fund


def vol_target(W, R, tv=0.40, cap=3.0, cost=COST, F=None, lb=30, funding_annual=0.11) -> pd.DataFrame:
    base = simulate(W, R, cost, F, funding_annual)
    rv = base.rolling(lb, min_periods=lb // 2).std() * np.sqrt(BPY)
    scale = (tv / rv).shift(1).clip(upper=cap).fillna(0.0)
    return W.mul(scale, axis=0)


def run(W, R, cost=COST, F=None, tv=0.40, funding_annual=0.11):
    Wt = vol_target(W, R, tv, 3.0, cost, F, funding_annual=funding_annual)
    return simulate(Wt, R, cost, F, funding_annual), Wt


def stats(net: pd.Series) -> dict:
    net = net.dropna()
    if len(net) < 30 or net.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, mdd=np.nan, vol=np.nan)
    eq = (1 + net).cumprod()
    return dict(sharpe=float(net.mean() / net.std() * np.sqrt(BPY)),
                cagr=float(eq.iloc[-1] ** (BPY / len(net)) - 1),
                mdd=float((eq / eq.cummax() - 1).min()), vol=float(net.std() * np.sqrt(BPY)))


def split(net: pd.Series, start=START):
    return net.loc[start:IS_END], net.loc[IS_END + pd.Timedelta(days=1):DESIGN_END]


def fmt(name, net, start=START) -> str:
    a, b = (stats(x) for x in split(net, start))
    return (f"{name:<44} IS Sharpe {a['sharpe']:5.2f} CAGR {a['cagr']:7.1%} DD {a['mdd']:6.1%} | "
            f"VAL Sharpe {b['sharpe']:5.2f} CAGR {b['cagr']:7.1%} DD {b['mdd']:6.1%}")


# ------------------------------------------------------------------------------------- panel
def load_panel(fresh: bool = False, min_qv: float = 5e6) -> dict:
    if WEEKEND_THR is None:
        raise RuntimeError("WEEKEND_THR not frozen yet: run v5_classifier.py first")
    c = vp.load("close", fresh)
    qv = vp.load("quote_volume", fresh)
    crypto = vp.crypto_mask(c, WEEKEND_THR)
    p = dict(c=c, qv=qv, v=vp.load("volume", fresh), tb=vp.load("taker_buy_quote_volume", fresh),
             F=vp.load("funding", fresh).reindex_like(c), crypto=crypto, r=c.pct_change(fill_method=None))
    p["e"] = vp.eligibility(c, qv, crypto, 60, min_qv, 30)
    return p


# ------------------------------------------------------------------------------------ sleeves
def v3_weights(c, e, signal_hook=None) -> pd.DataFrame:
    """v3 trend book. signal_hook(sig) may edit the raw [-1, 1] trend signal (used by B5 only)."""
    ret = c.pct_change(fill_method=None)
    sig = st.strength_signal(c, lookbacks=LBS, strength_vol=30, k=KSTEEP)
    if signal_hook is not None:
        sig = signal_hook(sig)
    raw = (sig * st._inv_vol(ret, 15)).where(e, 0.0)
    w6 = st.concentrate(st._normalize(raw), K_SELECT)
    return st.concentrate(st.ema_ensemble(w6, SPANS), K_HOLD)


def pump_weights(c, r, e, thr=0.25, hold=5, hedge="BTC"):
    idio = r.sub(r[hedge], axis=0)
    trig = (idio > thr) & e
    held = trig.astype(float).rolling(hold, min_periods=1).max().astype(bool) & e
    vol = r.rolling(20, min_periods=5).std()
    raw = (1.0 / vol).where(held, 0.0).fillna(0.0)
    w = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w[hedge] = w[hedge] - w.sum(axis=1)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return w, held


def v4(p: dict, cost=COST, F="real", v3_hook=None):
    """Returns (blend net, v3 net, pump net). F='real' uses the panel's funding matrix;
    F='generic' reproduces the old assumptions (11%/yr drag; 50%/yr stress on pumped longs)."""
    c, r, e = p["c"], p["r"], p["e"]
    w3 = v3_weights(c, e, v3_hook)
    wp, held = pump_weights(c, r, e)
    if F == "real":
        n3, _ = run(w3, r, cost, p["F"])
        npump, _ = run(wp, r, cost, p["F"])
    else:
        n3, _ = run(w3, r, cost, None)
        g = 0.11 / BPY
        Fs = pd.DataFrame(-g, index=c.index, columns=c.columns).mask(held, 0.50 / BPY)
        npump, _ = run(wp, r, cost, Fs)
    return 0.7 * n3 + 0.3 * npump, n3, npump


def xs_sleeve(signal, p: dict, cost=COST, min_names=10, ema=3):
    """Cross-sectional sleeve exactly as in the protocol: rank to [-0.5, 0.5], times 1/vol(15d),
    long and short sides each 0.5 gross, 3-day EMA, vol target 40%."""
    e, r = p["e"], p["r"]
    s = signal.where(e)
    n = s.notna().sum(axis=1)
    rk = s.rank(axis=1, pct=True) - 0.5 - 0.5 / n.replace(0, np.nan).to_numpy()[:, None]
    raw = rk * st._inv_vol(r, 15)
    raw = raw.where(e)
    lg = raw.clip(lower=0)
    sh = (-raw).clip(lower=0)
    w = (0.5 * lg.div(lg.sum(axis=1).replace(0, np.nan), axis=0)
         - 0.5 * sh.div(sh.sum(axis=1).replace(0, np.nan), axis=0)).fillna(0.0)
    w = w.where(n >= min_names, 0.0)
    w = w.ewm(span=ema).mean()
    return run(w, r, cost, p["F"])


def z30(x: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    return (x - x.rolling(n, min_periods=n * 2 // 3).mean()) / x.rolling(n, min_periods=n * 2 // 3).std()
