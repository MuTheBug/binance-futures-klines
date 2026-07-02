"""APEX — disciplined push past ECHO v2. Every change must clear an
IN-SAMPLE gate (fence: lab.IS_END = 2024-12-31); OOS is read once for the
final assembly at the end and reported no matter what it says.

Candidates:
  D  OI x price interaction sleeve ("conviction vs unwind"): ride moves
     backed by rising open interest, fade moves on falling OI (all four
     quadrants). OI data covers ~42 symbols from 2021 (like sleeve C).
  E  2-day rebalance cadence (echo_boost's banked finding, applied here).
  F  shrunk risk-parity sleeve mixing (50% shrink to the fixed prior).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import altdata
import echo
import engine
import lab
import strategies as st

close, vol, ret, elig = lab.load_data("1d")


def xz(df):
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1).replace(0.0, np.nan), axis=0)


def oi_flow_sleeve(oi, *, beta_window=60, formations=(3, 5, 8), cap=0.10,
                   use_resid=True, smooth=(3, 5, 8), gross=1.0):
    """Sleeve D: sig = tanh(z[ret_F]) * tanh(z[dOI_F]) summed over F.
    Same-sign (conviction) -> ride; opposite-sign (unwind) -> fade."""
    base = echo.residuals(ret, elig, beta_window=beta_window) if use_resid \
        else ret.where(elig)
    doi = np.log(oi.clip(lower=1e-9)).diff()
    sig = 0.0
    for F in formations:
        zr = xz(base.rolling(F, min_periods=F).sum().where(elig))
        zo = xz(doi.rolling(F, min_periods=F).sum().where(elig))
        sig = sig + np.tanh(zr) * np.tanh(zo)
    sig = (sig / len(formations)).where(elig)
    sig = sig.sub(sig.mean(axis=1), axis=0).where(elig)      # dollar-neutral
    w = sig.div(sig.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    w = sum(w.ewm(span=s).mean() for s in smooth) / len(smooth)
    w = w.clip(-cap, cap)
    s = w.abs().sum(axis=1).replace(0.0, np.nan)
    return w.div(s, axis=0).mul(gross).fillna(0.0)


def net_of(w):
    wv = engine.vol_target(w, ret, "1d", target_vol=0.40, max_leverage=3.0)
    return engine.simulate(wv, ret, "1d")["net"]


def shp(net):
    i, o = lab.split(net)
    return engine.metrics(i, "1d")["Sharpe"], engine.metrics(o, "1d")["Sharpe"]


def hold_cadence(w, days):
    idx = np.arange(len(w))
    keep = idx % days == 0
    out = w.copy()
    out.iloc[~keep] = np.nan
    return out.ffill().fillna(0.0)


def rp_mix(sleeves, priors, span=90, shrink=0.5):
    """Shrunk risk parity on sleeve nets: w_i ∝ prior_i / trailing vol,
    50% shrunk back to the prior. Scale known at t-1 (shifted)."""
    nets = [engine.simulate(w, ret, "1d")["net"] for w in sleeves]
    vols = [n.rolling(span, min_periods=span // 2).std() for n in nets]
    med = pd.concat(vols, axis=1).median(axis=1)
    mixed = 0.0
    raw = [pd.Series(p, index=close.index) * (med / v).clip(0.5, 2.0)
           for p, v in zip(priors, vols)]
    tot = sum(raw)
    for w, r, p in zip(sleeves, raw, priors):
        wt = ((1 - shrink) * (r / tot) + shrink * p).shift(1).fillna(p)
        mixed = mixed + w.mul(wt, axis=0)
    return mixed


def main():
    # ---- baseline: ECHO v2 -------------------------------------------------
    sigT = st.strength_signal(close, lookbacks=(15, 30, 60, 90),
                              strength_vol=30, k=2.0)
    A = st.concentrate(st.ema_ensemble(
        st.signal_to_weights(sigT, ret, elig, 15), (5, 10, 15)), 10)
    B = echo.residual_continuation(close, ret, elig, beta_window=60,
                                   formations=(3, 5, 8), cap=0.10)
    ls = altdata.load_metric("toptrader_pos_ls", reindex_like=close)
    C = echo.positioning_sleeve(close, ret, elig, ls, fade=True, lag=1,
                                beta_window=60, cap=0.10)
    base = 0.40 * A + 0.40 * B + 0.20 * C
    bi, bo = shp(net_of(base))
    print(f"BASE  ECHO v2 40/40/20:                 IS {bi:.2f}   (OOS held out)")

    # ---- candidate D: OI x price interaction (IS gate) ---------------------
    oi = altdata.load_metric("oi", reindex_like=close)
    D_best = None
    for use_resid in (True, False):
        D = oi_flow_sleeve(oi, use_resid=use_resid)
        di, _ = shp(net_of(D))
        c4 = 0.35 * A + 0.35 * B + 0.15 * C + 0.15 * D
        i4, _ = shp(net_of(c4))
        tag = "resid" if use_resid else "raw"
        verdict = "ACCEPT" if i4 > bi + 0.03 else "reject"
        print(f"  D({tag}): standalone IS {di:.2f}; blend 35/35/15/15 IS {i4:.2f}  -> {verdict}")
        if verdict == "ACCEPT" and (D_best is None or i4 > D_best[0]):
            D_best = (i4, D)

    sleeves = [A, B, C] + ([D_best[1]] if D_best else [])
    priors = [0.35, 0.35, 0.15, 0.15] if D_best else [0.40, 0.40, 0.20]
    book = sum(w * p for w, p in zip(sleeves, priors))

    # ---- candidate E: cadence (IS gate) ------------------------------------
    ci_1, _ = shp(net_of(book))
    best_cad, best_i = 1, ci_1
    for cad in (2, 3):
        i_c, _ = shp(net_of(hold_cadence(book, cad)))
        print(f"  E: cadence {cad}d IS {i_c:.2f}  (daily {ci_1:.2f})")
        if i_c > best_i + 0.03:
            best_cad, best_i = cad, i_c
    if best_cad > 1:
        book = hold_cadence(book, best_cad)
    print(f"  E verdict: cadence={best_cad}d")

    # ---- candidate F: shrunk risk-parity mixing -- REJECTED ----------------
    # On daily cadence RP adds nothing (IS 1.66-1.67 vs fixed 1.67 across
    # shrink 0.25/0.5/0.75); its apparent +0.03 on top of the 3d cadence was
    # interaction noise. Phase-robustness: 3d cadence IS by phase =
    # [1.71, 1.71, 1.68] (stable, kept); 2d = [1.61, 1.82] (unstable).

    # ---- FINAL: the single OOS read ----------------------------------------
    final = net_of(book)
    fi_, fo_ = shp(final)
    full = engine.metrics(final, "1d")
    print("\n" + "=" * 78)
    print(f"APEX final:  IS {fi_:.2f}   OOS {fo_:.2f}   "
          f"full Sharpe {full['Sharpe']:.2f}  CAGR {full['CAGR'] * 100:.0f}%  "
          f"maxDD {full['maxDD'] * 100:.0f}%  PF {full['profit_factor']:.2f}  "
          f"win {full['win_rate'] * 100:.0f}%")
    print(f"BASE  v2  :  IS {bi:.2f}   OOS {shp(net_of(base))[1]:.2f}")
    final.to_csv("results/apex_net.csv")
    eq = (1 + final.fillna(0)).cumprod()
    print("yearly:", {y: f"{v:+.0%}" for y, v in
                      eq.groupby(eq.index.year).agg(lambda x: x.iloc[-1] / x.iloc[0] - 1).items()})


if __name__ == "__main__":
    main()
