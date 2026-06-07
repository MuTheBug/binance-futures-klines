"""Full validation of v3 (strength top-6 + EMA-span-ensemble smoothing) + leverage/$60.
v3 = ema_ensemble(concentrate(ts_trend_strength)). Confirm robustness; honest on 2.0."""
from __future__ import annotations
import numpy as np, pandas as pd
import data, engine, strategies as st, lab, target as tgt

I = "1d"; TV = 0.40


def v3_weights(c, v, e, *, lookbacks=(15, 30, 60, 90), k=2.0, kcat=6, vlb=15, spans=(5, 10, 15)):
    w = st.ts_trend_strength(c, v, lookbacks=lookbacks, vol_lookback=vlb, k=k, elig=e)
    return st.ema_ensemble(st.concentrate(w, kcat), spans)


R = None  # set in main


def net_of(w, cost=engine.COST_PER_SIDE):
    return engine.simulate(engine.vol_target(w, R, I, target_vol=TV, cost_per_side=cost), R, I, cost_per_side=cost)["net"]


def L(tag, net):
    f, mi, mo = engine.metrics(net, I), engine.metrics(lab.split(net)[0], I), engine.metrics(lab.split(net)[1], I)
    print(f"  {tag:<26} FULL Sh={f['Sharpe']:>5.2f} | IS={mi['Sharpe']:>5.2f} | OOS={mo['Sharpe']:>5.2f} "
          f"| CAGR={f['CAGR']*100:>5.0f}% Cal={f['Calmar']:>4.2f} DD={f['maxDD']*100:>4.0f}%")
    return f, mi, mo


def main():
    global R
    c, v, r, e = lab.load_data(I)
    R = r

    print("="*98); print("V3 robustness gauntlet (must hold IS & OOS, show plateaus)"); print("="*98)
    L("v3 chosen", net_of(v3_weights(c, v, e)))

    print("\n k (trend steepness):")
    for k in [1.0, 1.5, 2.0, 3.0]:
        L(f"  k={k}", net_of(v3_weights(c, v, e, k=k)))
    print(" lookback set:")
    for lbs in [(15, 30, 60, 90), (10, 20, 40, 80), (20, 40, 80, 160), (30, 60, 120)]:
        L(f"  {lbs}", net_of(v3_weights(c, v, e, lookbacks=lbs)))
    print(" concentration K:")
    for kc in [5, 6, 8, 10]:
        L(f"  top{kc}", net_of(v3_weights(c, v, e, kcat=kc)))
    print(" EMA-span sets:")
    for sp in [(5, 10, 15), (3, 6, 12), (5, 15, 25), (8, 16, 24)]:
        L(f"  spans{sp}", net_of(v3_weights(c, v, e, spans=sp)))
    print(" cost/side sensitivity:")
    for bps in [0, 7, 15, 25, 40]:
        L(f"  {bps}bps", net_of(v3_weights(c, v, e), cost=bps/1e4))

    print("\n" + "="*98); print("V3 chosen — tearsheet, yearly, monthly"); print("="*98)
    w = v3_weights(c, v, e); net = net_of(w)
    print(engine.fmt_metrics(engine.metrics(net, I, "v3 FULL")))
    eq = (1 + net.fillna(0)).cumprod(); yr = eq.resample("YE").last().pct_change(); yr.iloc[0] = eq.resample("YE").last().iloc[0]-1
    print("  yearly: " + "  ".join(f"{t.year}:{x*100:+.0f}%" for t, x in yr.dropna().items()))
    ms = engine.monthly_summary(net)
    print(f"  monthly: mean={ms['mean_monthly']*100:.1f}% median={ms['median_monthly']*100:.1f}% "
          f"%+={ms['pct_positive']*100:.0f}% best={ms['best']*100:.0f}% worst={ms['worst']*100:.0f}%")

    print("\n" + "="*98); print("LEVERAGE vs RUIN ($60, 1yr, block-bootstrap w/ liquidation)"); print("="*98)
    full_arr = net.dropna().to_numpy(); oos_arr = lab.split(net)[1].dropna().to_numpy(); is_arr = lab.split(net)[0].dropna().to_numpy()
    print(f"  {'lev':>4} | {'IS-base med/mo $60':>20} | {'FULL med/mo $60':>18} | {'OOS med/mo $60':>18} | P(ruin)")
    for k in [1.0, 1.5, 2.0, 3.0]:
        ri = tgt.block_bootstrap(is_arr, k, n_paths=2500); rf = tgt.block_bootstrap(full_arr, k, n_paths=2500); ro = tgt.block_bootstrap(oos_arr, k, n_paths=2500)
        print(f"  {k:>4.1f} | {ri['median_monthly']*100:>7.1f}%  ${60*ri['median_year_x']:>6.0f} | "
              f"{rf['median_monthly']*100:>6.1f}%  ${60*rf['median_year_x']:>5.0f} | "
              f"{ro['median_monthly']*100:>6.1f}%  ${60*ro['median_year_x']:>5.0f} | {rf['p_ruin']*100:.0f}%")

    # save artifacts
    (1 + net.fillna(0)).cumprod().rename("equity").to_csv("results/equity_v3.csv")
    engine.monthly_returns(net).rename("monthly_return").to_csv("results/monthly_v3.csv")
    today = w.iloc[-1]; today = today[today.abs() > 1e-6].sort_values(key=abs, ascending=False)
    today.rename("target_weight").to_csv("results/today_target_weights.csv")
    print(f"\n  today ({c.index[-1].date()}) gross {today.abs().sum()*100:.0f}% @1x: " +
          ", ".join(f"{'L' if x>0 else 'S'}{s}" for s, x in today.items()))
    print("  saved results/equity_v3.csv, monthly_v3.csv, today_target_weights.csv")


if __name__ == "__main__":
    main()
