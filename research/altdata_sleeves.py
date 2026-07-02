"""
Build candidate ALT-DATA sleeves and judge them with the SAME discipline as everything else:
accept ONLY if a sleeve raises the IN-SAMPLE Sharpe of the ECHO blend (A trend + B residual).
OOS is a confirmation, never the target.

Run this AFTER downloading + pushing the data (see altdata/README.md):
    cd research && python3 altdata_sleeves.py

Candidate hypotheses (each ~orthogonal to price-trend):
  OIΔ      change in open interest        (new-money regime; test follow & fade)
  TOPpos   top-trader long/short (pos)    ("smart money" lean -> follow)
  GLOBAL   global account long/short      (retail crowd -> fade)
  TAKER    taker buy/sell volume ratio    (aggressive flow -> follow & fade)
  FUND     funding rate                   (overheated longs pay -> fade carry)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lab, engine, echo, strategies as st, altdata

close, vol, ret, elig = lab.load_data("1d")
IS_END = lab.IS_END


def sleeve(signal, *, fade=False, lag=1, beta_window=60, cap=0.10, smooth=(3, 5, 8)):
    """Same construction as the price sleeves: tanh-bounded, dollar- & beta-neutral,
    inverse-vol, EMA-smoothed. `lag` shifts the alt-data by N days (no look-ahead)."""
    s = signal.reindex_like(close).shift(lag).where(elig)
    z = s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1).replace(0, np.nan), axis=0)
    sig = (-np.tanh(z) if fade else np.tanh(z)).where(elig)
    sig = sig.sub(sig.mean(axis=1), axis=0).where(elig)
    beta = echo.rolling_beta(ret, echo.market_return(ret, elig), beta_window).where(elig, 0.0)
    w = (sig * (1.0 / ret.rolling(30, min_periods=15).std().replace(0, np.nan))).where(elig, 0.0)
    num = (w * beta).sum(axis=1); den = (beta * beta).sum(axis=1).replace(0.0, np.nan)
    w = w.sub(beta.mul(num / den, axis=0), fill_value=0.0).where(elig, 0.0)
    w = st.ema_ensemble(w, smooth).clip(-cap, cap)
    return w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def net_of(w):
    return engine.simulate(engine.vol_target(w, ret, "1d", target_vol=0.40, max_leverage=3.0), ret, "1d")["net"]


def shp(net):
    i, o = lab.split(net); return engine.metrics(i, "1d")["Sharpe"], engine.metrics(o, "1d")["Sharpe"]


def main():
    if not altdata.have_metrics() and not altdata.have_funding():
        print("No alt-data found in ../altdata/. Run altdata/fetch_metrics.py and "
              "fetch_funding.py first, then push (see altdata/README.md).")
        return

    # ECHO baseline
    sigT = st.strength_signal(close, lookbacks=(15, 30, 60, 90), strength_vol=30, k=2.0)
    A = st.concentrate(st.ema_ensemble(st.signal_to_weights(sigT, ret, elig, 15), (5, 10, 15)), 10)
    B = echo.residual_continuation(close, ret, elig, beta_window=60, formations=(3, 5, 8), cap=0.10)
    nA, nB = net_of(A), net_of(B)
    base = net_of(0.5 * A + 0.5 * B)
    base_is = engine.metrics(lab.split(base)[0], "1d")["Sharpe"]
    print("=" * 92)
    print(f"ECHO baseline (A trend + B residual, 50/50): IS Sharpe {base_is:.2f}")
    print("=" * 92)

    # assemble candidate raw signals from whatever data is present
    cands = {}
    oi = altdata.load_metric("oi", reindex_like=close) if altdata.have_metrics() else None
    if oi is not None:
        cands["OIΔ  d_log(OI)"] = np.log(oi.clip(lower=1e-9)).diff()
    for fld, nm in [("toptrader_pos_ls", "TOPpos top-trader L/S"),
                    ("global_acc_ls", "GLOBAL retail L/S"),
                    ("taker_ls", "TAKER buy/sell")]:
        if altdata.have_metrics():
            m = altdata.load_metric(fld, reindex_like=close)
            if m is not None and m.notna().sum().sum() > 1000:
                cands[nm] = m
    if altdata.have_funding():
        fnd = altdata.load_funding("funding_sum", reindex_like=close)
        if fnd is not None:
            cands["FUND funding rate"] = fnd

    if not cands:
        print("Alt-data files present but no usable columns / too sparse.")
        return

    print(f"{'candidate':24} {'dir':5} {'cov%':>5} {'IS':>5} {'OOS':>6} | "
          f"{'corrA':>6} {'corrB':>6} | {'blend+ IS':>9} {'OOS':>6} {'verdict':>8}")
    accepted = {}
    for nm, raw in cands.items():
        cov = float(raw.reindex_like(close).notna().mean().mean()) * 100
        best = None
        for fade in (False, True):
            w = sleeve(raw, fade=fade)
            nn = net_of(w); i, o = shp(nn)
            if best is None or i > best[0]:
                best = (i, o, fade, w, nn)
        i, o, fade, w, nn = best
        cdf = pd.concat([lab.split(nA)[0], lab.split(nB)[0], lab.split(nn)[0]], axis=1).dropna()
        cA = cdf.iloc[:, 0].corr(cdf.iloc[:, 2]); cB = cdf.iloc[:, 1].corr(cdf.iloc[:, 2])
        three = net_of((A + B + w) / 3.0); ti, to = shp(three)
        verdict = "ACCEPT" if ti > base_is + 0.03 else "reject"
        if verdict == "ACCEPT":
            accepted[nm] = w
        print(f"{nm:24} {'fade' if fade else 'ride':5} {cov:5.0f} {i:5.2f} {o:6.2f} | "
              f"{cA:6.2f} {cB:6.2f} | {ti:9.2f} {to:6.2f} {verdict:>8}")

    if accepted:
        sleeves = [A, B] + list(accepted.values())
        combo = sum(sleeves) / len(sleeves)
        ci, co = shp(net_of(combo))
        print("\n" + "=" * 92)
        print(f"FULL blend A+B+{'+'.join(k.split()[0] for k in accepted)}: "
              f"IS {ci:.2f} | OOS {co:.2f}   (baseline IS {base_is:.2f})")
        print("If IS rose materially and held OOS, this is a real new edge -> wire into echo_engine/bot.")
    else:
        print("\nNo alt-data sleeve raised in-sample Sharpe past the bar -> honest reject. "
              "(Coverage/history may be too short; re-run after collecting more.)")


if __name__ == "__main__":
    main()
