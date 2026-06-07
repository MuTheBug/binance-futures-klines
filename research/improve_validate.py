"""Full robustness gauntlet for the IMPROVED flagship (risk-adjusted trend strength),
plus refreshed leverage/ruin. We must see plateaus + positive IS AND OOS, like before."""
from __future__ import annotations
import numpy as np
import pandas as pd
import data, engine, strategies as st
import lab, target as tgt

INTERVAL = "1d"; TV = 0.40; K = 8
pd.set_option("display.width", 220)


def run(close, vol, ret, elig, *, lookbacks=(15, 30, 60, 90), k=2.0, vlb=15, kcat=K,
        cost=engine.COST_PER_SIDE, eligm=None):
    em = elig if eligm is None else eligm
    w = st.ts_trend_strength(close, vol, lookbacks=lookbacks, vol_lookback=vlb, k=k, elig=em)
    w = st.concentrate(w, kcat)
    w = engine.vol_target(w, ret, INTERVAL, target_vol=TV, cost_per_side=cost)
    net = engine.simulate(w, ret, INTERVAL, cost_per_side=cost)["net"]
    return net, w


def IO(net):
    mi, mo = engine.metrics(lab.split(net)[0], INTERVAL), engine.metrics(lab.split(net)[1], INTERVAL)
    return mi, mo


def line(tag, net):
    mi, mo = IO(net)
    print(f"  {tag:<26} IS Sh={mi['Sharpe']:>5.2f} CAGR={mi['CAGR']*100:>6.1f}% DD={mi['maxDD']*100:>5.0f}% | "
          f"OOS Sh={mo['Sharpe']:>5.2f} CAGR={mo['CAGR']*100:>6.1f}% DD={mo['maxDD']*100:>5.0f}%")
    return mi, mo


def main():
    c, v, r, e = lab.load_data(INTERVAL)

    print("="*100); print("A) k (trend-strength steepness) plateau — want a broad stable region"); print("="*100)
    for k in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        line(f"k={k}", run(c, v, r, e, k=k)[0])

    print("\n" + "="*100); print("B) lookback-set robustness (k=2)"); print("="*100)
    for lbs in [(15, 30, 60, 90), (10, 20, 40, 80), (20, 40, 80, 160), (15, 45, 90), (30, 60, 120), (10, 30, 60, 120, 180)]:
        line(f"{lbs}", run(c, v, r, e, lookbacks=lbs)[0])

    print("\n" + "="*100); print("C) cost sensitivity (k=2)"); print("="*100)
    for bps in [0, 7, 15, 25, 40]:
        line(f"cost/side={bps}bps", run(c, v, r, e, cost=bps/1e4)[0])

    print("\n" + "="*100); print("D) concentration K (runnable on $60 = small K)"); print("="*100)
    for kc in [4, 5, 6, 8, 10, 15, 9999]:
        line(f"top{kc if kc<9999 else 'ALL'}", run(c, v, r, e, kcat=kc)[0])

    print("\n" + "="*100); print("E) universe breadth (top-N liquid, point-in-time, k=2)"); print("="*100)
    dollar = (c * v); liq_rank = dollar.rolling(30, min_periods=15).median().rank(axis=1, ascending=False)
    for N in [10, 20, 30, 50, 9999]:
        line(f"univ top{N if N<9999 else 'ALL'}", run(c, v, r, e, eligm=e & (liq_rank <= N))[0])

    print("\n" + "="*100); print("F) chosen config: top-8, k=2 — full tearsheet + yearly"); print("="*100)
    net, w = run(c, v, r, e)
    full = engine.metrics(net, INTERVAL, "improved top8 FULL")
    print(engine.fmt_metrics(full));
    mi, mo = line("improved top8", net)
    eq = (1 + net.fillna(0)).cumprod(); yr = eq.resample("YE").last().pct_change(); yr.iloc[0] = eq.resample("YE").last().iloc[0]-1
    print("   yearly: " + "  ".join(f"{t.year}:{x*100:+.0f}%" for t, x in yr.dropna().items()))
    ms = engine.monthly_summary(net)
    print(f"   monthly mean={ms['mean_monthly']*100:.1f}% median={ms['median_monthly']*100:.1f}% "
          f"%+={ms['pct_positive']*100:.0f}% best={ms['best']*100:.0f}% worst={ms['worst']*100:.0f}%")

    print("\n" + "="*100); print("G) LEVERAGE vs RUIN (improved top-8), 1yr block-bootstrap w/ liquidation"); print("="*100)
    full_arr = net.dropna().to_numpy(); oos_arr = lab.split(net)[1].dropna().to_numpy()
    print(f"  {'lev':>4} | {'FULL: med.mo $60->1yr P(ruin)':>34} | {'OOS: med.mo $60->1yr P(ruin)':>34}")
    for k in [1.0, 1.5, 2.0, 3.0, 5.0]:
        rf = tgt.block_bootstrap(full_arr, k, n_paths=3000); ro = tgt.block_bootstrap(oos_arr, k, n_paths=3000)
        print(f"  {k:>4.1f} | {rf['median_monthly']*100:>7.1f}%  ${60*rf['median_year_x']:>6.0f}  {rf['p_ruin']*100:>5.0f}% "
              f"        | {ro['median_monthly']*100:>7.1f}%  ${60*ro['median_year_x']:>6.0f}  {ro['p_ruin']*100:>5.0f}%")

    # save artifacts
    (1 + net.fillna(0)).cumprod().rename("equity").to_csv("results/equity_improved_top8.csv")
    engine.monthly_returns(net).rename("monthly_return").to_csv("results/monthly_improved_top8.csv")
    today = w.iloc[-1]; today = today[today.abs() > 1e-6].sort_values(key=abs, ascending=False)
    today.rename("target_weight").to_csv("results/today_target_weights_improved.csv")
    print("\n  saved improved equity/monthly/today-weights to results/")


if __name__ == "__main__":
    main()
