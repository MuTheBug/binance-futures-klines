"""Funding-carry sleeve (cross-sectional, market-neutral): long negative-funding,
short high-funding coins. Test standalone, measure correlation to trend, then combine."""
from __future__ import annotations
import numpy as np, pandas as pd
import engine, strategies as st, lab, funding
import final_v3 as fv

I = "1d"; TV = 0.40


def xs_z(df, elig):
    d = df.where(elig)
    return d.sub(d.mean(axis=1), axis=0).div(d.std(axis=1).replace(0, np.nan), axis=0)


FMAT = None   # set in main: per-coin daily funding matrix (real harvest income)


def sim_net(w, r):
    wv = engine.vol_target(w, r, I, target_vol=TV, funding_matrix=FMAT)
    return engine.simulate(wv, r, I, funding_matrix=FMAT)["net"]


def io(net, tag, base=None):
    f, mi, mo = engine.metrics(net, I), engine.metrics(lab.split(net)[0], I), engine.metrics(lab.split(net)[1], I)
    extra = ""
    if base is not None:
        cc = pd.concat([net, base], axis=1).dropna(); extra = f"  corr2trend={cc.iloc[:,0].corr(cc.iloc[:,1]):+.2f}"
    print(f"  {tag:<26} FULL={f['Sharpe']:>5.2f} IS={mi['Sharpe']:>5.2f} OOS={mo['Sharpe']:>5.2f} "
          f"CAGR={f['CAGR']*100:>5.0f}% DD={f['maxDD']*100:>4.0f}%{extra}")
    return net


def carry_weights(F, r, e, N=3, K=None, smooth=10):
    Fbar = F.rolling(N, min_periods=max(1, N // 2)).mean()
    sig = -np.tanh(xs_z(Fbar, e))                       # short high funding / long low
    w = st.signal_to_weights(sig.reindex_like(r).fillna(0.0), r, e, vol_lookback=15)
    if K:
        w = st.concentrate(w, K)
    if smooth:
        w = st.ema_ensemble(w, (5, 10, 15)) if smooth == "ens" else w.ewm(span=smooth).mean()
        if K:
            w = st.concentrate(w, K)
    return w


def main():
    global FMAT
    c, v, r, e = lab.load_data(I)
    e = e & ~e.columns.isin(fv.EXCLUDE)                 # crypto-only
    F = funding.load_daily_funding(c.index).reindex(columns=c.columns)
    e = e & F.notna()                                   # only trade coins with funding data
    FMAT = F                                            # real per-coin funding income/cost

    trend = sim_net(fv.build(c, v, e), r)
    print("="*100); print("TREND sleeve (v3, crypto-only, on funding-covered universe)"); print("="*100)
    io(trend, "trend v3")

    print("\n" + "="*100); print("CARRY sleeve standalone — funding lookback N, full book"); print("="*100)
    for N in [1, 3, 5, 7, 10]:
        io(sim_net(carry_weights(F, r, e, N=N, K=None, smooth=0), r), f"carry N={N} (raw)", trend)

    print("\n carry with EMA10 smoothing + concentration (runnable):")
    for N in [3, 5]:
        for K in [None, 12, 8]:
            io(sim_net(carry_weights(F, r, e, N=N, K=K, smooth=10), r), f"carry N={N} K={K} EMA10", trend)

    print("\n" + "="*100); print("COMBINE trend + carry (net-level blend; Sharpe is scale-free)"); print("="*100)
    carry = sim_net(carry_weights(F, r, e, N=3, K=12, smooth=10), r)
    for wt in [0.2, 0.3, 0.4, 0.5]:
        combo = ((1 - wt) * trend + wt * carry).dropna()
        io(combo, f"trend{int((1-wt)*100)}/carry{int(wt*100)}")
    # inverse-vol optimal-ish blend
    cc = pd.concat([trend, carry], axis=1).dropna(); cc.columns = ["t", "c"]
    rho = cc.t.corr(cc.c)
    print(f"\n  correlation(trend,carry) = {rho:+.2f}  -> diversification ratio favors combining if low")


if __name__ == "__main__":
    main()
