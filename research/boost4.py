"""Runnability check: the EMA-smoothed book holds ~37 tiny names (un-runnable on $60).
Realistic version = smooth then keep only executable names (top-K). Does Sharpe survive?"""
from __future__ import annotations
import numpy as np, pandas as pd
import data, engine, strategies as st, lab

I = "1d"; TV = 0.40


def sim(w, r):
    s = engine.simulate(engine.vol_target(w, r, I, target_vol=TV), r, I)
    net = s["net"]
    f, mi, mo = engine.metrics(net, I), engine.metrics(lab.split(net)[0], I), engine.metrics(lab.split(net)[1], I)
    npos = (w.abs() > 1e-4).sum(axis=1)
    npos = npos[npos > 0]
    print(f"  FULL={f['Sharpe']:>5.2f} IS={mi['Sharpe']:>5.2f} OOS={mo['Sharpe']:>5.2f} "
          f"Cal={f['Calmar']:>4.2f} DD={f['maxDD']*100:>4.0f}% turn={s['turnover'].mean():.2f} "
          f"#pos~{npos.median():.0f}")
    return net


def main():
    c, v, r, e = lab.load_data(I)
    strength = st.strength_signal(c, lookbacks=(15, 30, 60, 90), k=2.0)

    print("="*96)
    print("Reference"); print("="*96)
    print(" unsmoothed top6 (runnable, v2):", end=" "); sim(st.concentrate(st.ts_trend_strength(c, v, lookbacks=(15,30,60,90), vol_lookback=15, k=2.0, elig=e), 6), r)
    print(" smoothed top6->EMA (NOT runnable, 37 names):", end=" ")
    sim(st.ema_ensemble(st.concentrate(st.ts_trend_strength(c, v, lookbacks=(15,30,60,90), vol_lookback=15, k=2.0, elig=e), 6)), r)

    print("\n" + "="*96)
    print("RUNNABLE A: concentrate(6) -> EMA-smooth -> RE-concentrate to top-K (bounded positions)")
    print("="*96)
    w6 = st.concentrate(st.ts_trend_strength(c, v, lookbacks=(15,30,60,90), vol_lookback=15, k=2.0, elig=e), 6)
    sm = st.ema_ensemble(w6)
    for K in [6, 8, 10, 12]:
        print(f"  K={K:<3}", end=""); sim(st.concentrate(sm, K), r)

    print("\n" + "="*96)
    print("RUNNABLE B: smooth FULL signal-weights -> concentrate to top-K (stable set, low turnover)")
    print("="*96)
    wfull = st.signal_to_weights(strength, r, e, vol_lookback=15)
    smf = st.ema_ensemble(wfull)
    for K in [6, 8, 10, 12]:
        print(f"  K={K:<3}", end=""); sim(st.concentrate(smf, K), r)

    print("\n" + "="*96)
    print("RUNNABLE C: smooth the SIGNAL, size, concentrate (alt ordering)")
    print("="*96)
    for K in [6, 8, 10]:
        s_sm = st.ema_ensemble(strength.fillna(0.0))
        w = st.concentrate(st.signal_to_weights(s_sm, r, e, vol_lookback=15), K)
        print(f"  K={K:<3}", end=""); sim(w, r)


if __name__ == "__main__":
    main()
