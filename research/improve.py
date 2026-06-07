"""Honestly try to push Sharpe higher WITHIN the robust trend family (low overfit risk).
A real improvement must beat the flagship in BOTH IS and OOS, not just one."""
from __future__ import annotations
import numpy as np
import pandas as pd
import data, engine, strategies as st
import lab

INTERVAL = "1d"; TV = 0.40; K = 8
LBS = (15, 30, 60, 90)


def main():
    c, v, r, e = lab.load_data(INTERVAL)

    def evalsig(name, signal, vlb=15):
        w = st.signal_to_weights(signal.reindex_like(c).fillna(0.0), r, e, vol_lookback=vlb)
        w = st.concentrate(w, K)
        w = engine.vol_target(w, r, INTERVAL, target_vol=TV)
        net = engine.simulate(w, r, INTERVAL)["net"]
        mi, mo = engine.metrics(lab.split(net)[0], INTERVAL), engine.metrics(lab.split(net)[1], INTERVAL)
        flag = " <-- beats flagship in BOTH" if (mi['Sharpe'] > 1.38 and mo['Sharpe'] > 0.57) else ""
        print(f"  {name:<30} IS Sh={mi['Sharpe']:>5.2f} CAGR={mi['CAGR']*100:>6.1f}% | "
              f"OOS Sh={mo['Sharpe']:>5.2f} CAGR={mo['CAGR']*100:>6.1f}%{flag}")
        return net

    ret = c.pct_change()
    vol = ret.rolling(30, min_periods=15).std()

    # baseline
    sign_sig = sum(np.sign(c.pct_change(L)) for L in LBS) / len(LBS)
    evalsig("flagship sign-trend", sign_sig)

    # A) continuous risk-adjusted trend, squashed with tanh (weights stronger trends more)
    for kk in [1.0, 2.0, 3.0]:
        z = sum(np.tanh(kk * c.pct_change(L) / (vol * np.sqrt(L))) for L in LBS) / len(LBS)
        evalsig(f"tanh risk-adj trend k={kk}", z)

    # B) raw risk-adjusted (z-score) clipped
    z = sum((c.pct_change(L) / (vol * np.sqrt(L))).clip(-3, 3) for L in LBS) / len(LBS)
    evalsig("z-score trend (clip3)", z)

    # C) skip most-recent days (reduce 1-bar whipsaw): use close shifted by `skip`
    for skip in [1, 2, 3]:
        s = sum(np.sign(c.shift(skip).pct_change(L)) for L in LBS) / len(LBS)
        evalsig(f"sign-trend skip={skip}", s)

    # D) EMA-smoothed sign (reduce flip-flop)
    for span in [3, 5, 8]:
        s = (sum(np.sign(c.pct_change(L)) for L in LBS) / len(LBS)).ewm(span=span).mean()
        evalsig(f"sign-trend EMA{span}", s)

    # E) MA-distance trend (price vs moving average, normalized)
    s = sum(np.sign(c / c.rolling(L).mean() - 1.0) for L in LBS) / len(LBS)
    evalsig("price-vs-SMA trend", s)

    # F) longer vol_lookback for sizing (smoother risk weights)
    evalsig("flagship, vol_lb=30", sign_sig, vlb=30)
    evalsig("flagship, vol_lb=45", sign_sig, vlb=45)


if __name__ == "__main__":
    main()
