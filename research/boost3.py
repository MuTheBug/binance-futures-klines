"""Round 3: nail the best ROBUST combo. Test the missing cells (gk+EMA, blend+cc+EMA,
ensemble-of-EMA-spans), with turnover & cost diagnostics. Honest about IS vs OOS gap."""
from __future__ import annotations
import numpy as np, pandas as pd
import data, engine, strategies as st, lab
from boost2 import gk_sigma, sig_to_w, blend_signal

I = "1d"; TV = 0.40


def vt_w(w, r, target=TV):
    return engine.vol_target(w, r, I, target_vol=target)


def report(tag, w, r, base=None, costs=False):
    sim = engine.simulate(vt_w(w, r), r, I)
    net = sim["net"]
    f, mi, mo = engine.metrics(net, I), engine.metrics(lab.split(net)[0], I), engine.metrics(lab.split(net)[1], I)
    extra = ""
    if base is not None:
        cc = pd.concat([net, base], axis=1).dropna(); extra = f" corr={cc.iloc[:,0].corr(cc.iloc[:,1]):.2f}"
    to = sim["turnover"].mean()
    print(f"  {tag:<30} FULL={f['Sharpe']:>5.2f} IS={mi['Sharpe']:>5.2f} OOS={mo['Sharpe']:>5.2f} "
          f"Cal={f['Calmar']:>4.2f} DD={f['maxDD']*100:>4.0f}% turn={to:.2f}{extra}")
    return net


def main():
    c, v, r, e = lab.load_data(I)
    cc_sigma = r.rolling(15, min_periods=8).std()
    gk = gk_sigma(15, c)
    str_sig = st.strength_signal(c, lookbacks=(15, 30, 60, 90), k=2.0)
    bs = blend_signal(c)

    print("="*104); print("Reference"); print("="*104)
    base = report("baseline (str,cc,top6)", st.concentrate(sig_to_w(str_sig, cc_sigma, e), 6), r)

    print("\n" + "="*104); print("A) strength + GK-vol + weightEMA  (gk improved IS; EMA improves both)"); print("="*104)
    w_gk = st.concentrate(sig_to_w(str_sig, gk, e), 6)
    for N in [8, 10, 12]:
        report(f"str+gk+EMA{N}", w_gk.ewm(span=N).mean(), r, base)

    print("\n" + "="*104); print("B) blend + CC-vol + weightEMA  (the two round-1 winners, no gk)"); print("="*104)
    w_b = st.concentrate(sig_to_w(bs, cc_sigma, e), 6)
    for N in [8, 10, 12]:
        report(f"blend+cc+EMA{N}", w_b.ewm(span=N).mean(), r, base)

    print("\n" + "="*104); print("C) ENSEMBLE of EMA spans (avg of EMA5,10,15 weights) — avoid picking one span"); print("="*104)
    for name, W in [("str+cc", st.concentrate(sig_to_w(str_sig, cc_sigma, e), 6)),
                    ("str+gk", w_gk), ("blend+cc", w_b)]:
        ens = (W.ewm(span=5).mean() + W.ewm(span=10).mean() + W.ewm(span=15).mean()) / 3
        report(f"{name}+EMAens", ens, r, base)

    print("\n" + "="*104); print("D) best candidate -> yearly + cost sensitivity"); print("="*104)
    best = w_gk.ewm(span=10).mean()    # strength + gk-vol + EMA10
    net = report("CANDIDATE str+gk+EMA10", best, r, base)
    eq = (1 + net.fillna(0)).cumprod(); yr = eq.resample("YE").last().pct_change(); yr.iloc[0] = eq.resample("YE").last().iloc[0]-1
    print("   yearly: " + "  ".join(f"{t.year}:{x*100:+.0f}%" for t, x in yr.dropna().items()))
    print("   cost sensitivity (per side):")
    for bps in [0, 7, 15, 25, 40]:
        s = engine.simulate(vt_w(best, r), r, I, cost_per_side=bps/1e4)["net"]
        mi, mo = engine.metrics(lab.split(s)[0], I), engine.metrics(lab.split(s)[1], I)
        print(f"      {bps:>3}bps: IS Sh={mi['Sharpe']:.2f}  OOS Sh={mo['Sharpe']:.2f}")


if __name__ == "__main__":
    main()
