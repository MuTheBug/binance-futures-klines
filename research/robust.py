"""
Anti-overfit robustness suite — ALL computed on IN-SAMPLE data only (<= IS_END).
We want PLATEAUS not spikes: the edge should survive parameter changes, higher
costs, a smaller universe, and should be spread across many symbols & sub-periods.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import data, engine, strategies as st
import lab

INTERVAL = "1d"
pd.set_option("display.width", 200)


def is_only(s):
    return s[s.index <= lab.IS_END]


def is_sharpe(weights, ret, cost=engine.COST_PER_SIDE):
    w = engine.vol_target(weights, ret, INTERVAL, target_vol=0.40, cost_per_side=cost)
    net = is_only(engine.simulate(w, ret, INTERVAL, cost_per_side=cost)["net"])
    m = engine.metrics(net, INTERVAL)
    return m["Sharpe"], m["CAGR"], m["maxDD"], m["Calmar"]


def main():
    close, vol, ret, elig = lab.load_data(INTERVAL)
    print(f"IS window: {close.index[0].date()} .. {lab.IS_END.date()}  universe={close.shape[1]}\n")

    # ---------- A) lookback x vol_lookback heatmap (single-lookback L/S) ----------
    print("="*90)
    print("A) IS Sharpe heatmap: single-lookback L/S trend  (rows=trend LB days, cols=vol LB days)")
    print("   Robust = a broad region of positive Sharpe, not one lonely cell.")
    lookbacks = [10, 15, 20, 30, 45, 60, 90, 120, 180]
    vlbs = [15, 30, 45, 60]
    tbl = pd.DataFrame(index=lookbacks, columns=vlbs, dtype=float)
    for L in lookbacks:
        for V in vlbs:
            w = st.ts_momentum(close, vol, lookback=L, vol_lookback=V, elig=elig)
            tbl.loc[L, V] = is_sharpe(w, ret)[0]
    print(tbl.round(2).to_string())
    print(f"  -> fraction of grid with Sharpe>0.8: {(tbl>0.8).mean().mean():.0%}; "
          f"min={tbl.min().min():.2f} max={tbl.max().max():.2f} median={tbl.values.flatten()[~np.isnan(tbl.values.flatten())].__class__ and np.nanmedian(tbl.values):.2f}")

    # ---------- B) multi-lookback set robustness ----------
    print("\n" + "="*90)
    print("B) IS metrics for various multi-lookback SETS (L/S, vt40). Different sensible")
    print("   sets should give similar results if the edge is real (not set-specific).")
    sets = {
        "(15,30,60,90) chosen": (15, 30, 60, 90),
        "(10,20,40,80)":        (10, 20, 40, 80),
        "(20,40,80,160)":       (20, 40, 80, 160),
        "(15,45,90)":           (15, 45, 90),
        "(30,60,120)":          (30, 60, 120),
        "(10,30,60,120,180)":   (10, 30, 60, 120, 180),
    }
    for name, lbs in sets.items():
        w = st.ts_momentum_multi(close, vol, lookbacks=lbs, vol_lookback=lbs[0], elig=elig)
        s, c, dd, cal = is_sharpe(w, ret)
        print(f"   {name:>26}  Sharpe={s:>5.2f}  CAGR={c*100:>6.1f}%  maxDD={dd*100:>6.1f}%  Calmar={cal:>5.2f}")

    # ---------- C) cost sensitivity ----------
    print("\n" + "="*90)
    print("C) Cost sensitivity (chosen multi-set). Edge must survive realistic+pessimistic costs.")
    w_chosen = st.ts_momentum_multi(close, vol, lookbacks=(15, 30, 60, 90), vol_lookback=15, elig=elig)
    for bps in [0, 5, 7, 10, 15, 25, 40]:
        s, c, dd, cal = is_sharpe(w_chosen, ret, cost=bps/1e4)
        print(f"   cost/side={bps:>3}bps (round-trip {2*bps:>3}bps)  Sharpe={s:>5.2f}  CAGR={c*100:>6.1f}%  Calmar={cal:>5.2f}")

    # ---------- D) universe breadth: does it depend on a few coins? ----------
    print("\n" + "="*90)
    print("D) Universe / breadth sensitivity. Restrict to top-N most-liquid names (point-in-time).")
    dollar = (close * vol)
    liq_rank = dollar.rolling(30, min_periods=15).median().rank(axis=1, ascending=False)
    for N in [10, 20, 30, 50, 9999]:
        elig_n = elig & (liq_rank <= N)
        w = st.ts_momentum_multi(close, vol, lookbacks=(15, 30, 60, 90), vol_lookback=15, elig=elig_n)
        s, c, dd, cal = is_sharpe(w, ret)
        lab_n = "ALL" if N == 9999 else f"top{N}"
        print(f"   universe={lab_n:>6}  Sharpe={s:>5.2f}  CAGR={c*100:>6.1f}%  maxDD={dd*100:>6.1f}%  Calmar={cal:>5.2f}")

    # ---------- E) sub-period stability (full sample, rolling 90-bar Sharpe) ----------
    print("\n" + "="*90)
    print("E) Sub-period stability of chosen config across the FULL sample (incl. OOS).")
    print("   We want positive Sharpe in most sub-periods, not one giant lucky run.")
    w = engine.vol_target(w_chosen, ret, INTERVAL, target_vol=0.40)
    net = engine.simulate(w, ret, INTERVAL)["net"]
    # calendar-year returns
    eq = (1 + net.fillna(0)).cumprod()
    yr = eq.resample("YE").last().pct_change()
    yr.iloc[0] = eq.resample("YE").last().iloc[0] - 1
    print("   calendar-year returns:")
    for ts, v in yr.dropna().items():
        print(f"      {ts.year}: {v*100:>7.1f}%")
    roll = net.rolling(90).mean() / net.rolling(90).std() * np.sqrt(365)
    roll = roll.dropna()
    print(f"   rolling 90d Sharpe: pct of time >0: {(roll>0).mean():.0%}; "
          f"median={roll.median():.2f}; p10={roll.quantile(.1):.2f}; p90={roll.quantile(.9):.2f}")


if __name__ == "__main__":
    main()
