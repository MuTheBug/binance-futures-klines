"""
Final tearsheet for the recommended strategy + honest $60 projection.
Strategy: DAILY, multi-lookback (15/30/60/90d) time-series momentum, LONG & SHORT,
inverse-vol sized, portfolio vol-targeted to ~40% ann. Full book and top-8 (runnable).
Saves equity curve, monthly table, and current-day target weights to results/.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import data, engine, strategies as st
import lab

INTERVAL = "1d"
LBS = (15, 30, 60, 90)
TV = 0.40


def build(k=None):
    c, v, r, e = lab.load_data(INTERVAL)
    w = st.ts_momentum_multi(c, v, lookbacks=LBS, vol_lookback=LBS[0], elig=e)
    if k:
        w = st.concentrate(w, k)
    wv = engine.vol_target(w, r, INTERVAL, target_vol=TV)
    sim = engine.simulate(wv, r, INTERVAL)
    return c, v, r, e, wv, sim


def tearsheet(name, net):
    full = engine.metrics(net, INTERVAL, name + " FULL")
    is_, oos_ = lab.split(net)
    print(engine.fmt_metrics(full))
    print(engine.fmt_metrics(engine.metrics(is_, INTERVAL, name + " IS ")))
    print(engine.fmt_metrics(engine.metrics(oos_, INTERVAL, name + " OOS")))
    return full


def main():
    print("#"*100)
    print("RECOMMENDED STRATEGY — Daily multi-lookback L/S trend, vol-targeted")
    print("#"*100, "\n")

    c, v, r, e, wv_full, sim_full = build(k=None)
    c8, v8, r8, e8, wv8, sim8 = build(k=8)
    net_full, net8 = sim_full["net"], sim8["net"]

    print(">> FULL BOOK (ideal, needs many positions):")
    tearsheet("full-book", net_full)
    print("\n>> TOP-8 (runnable on a small account):")
    m8 = tearsheet("top8", net8)

    # yearly
    print("\nCalendar-year returns (top-8, gross=vol-targeted ~1x, NO extra leverage):")
    eq = (1 + net8.fillna(0)).cumprod()
    yr = eq.resample("YE").last().pct_change(); yr.iloc[0] = eq.resample("YE").last().iloc[0] - 1
    print("   " + "  ".join(f"{ts.year}:{v*100:+.0f}%" for ts, v in yr.dropna().items()))

    # recent monthly (last 18)
    m = engine.monthly_returns(net8)
    print("\nLast 18 monthly returns (top-8, ~1x):")
    print("   " + "  ".join(f"{ts.strftime('%y-%m')}:{x*100:+.0f}%" for ts, x in m.tail(18).items()))
    ms = engine.monthly_summary(net8)
    print(f"   monthly mean={ms['mean_monthly']*100:.1f}% median={ms['median_monthly']*100:.1f}% "
          f"%+={ms['pct_positive']*100:.0f}% best={ms['best']*100:.0f}% worst={ms['worst']*100:.0f}%")

    # ---- honest $60 projection at prudent leverage ----
    print("\n" + "="*100)
    print("HONEST $60 PROJECTION (top-8). 'med.month' from block-bootstrap of daily P&L.")
    print("="*100)
    import target as tgt
    full_arr = net8.dropna().to_numpy()
    oos_arr = lab.split(net8)[1].dropna().to_numpy()
    print(f"  {'lev':>4} | {'FULL-sample base':^34} | {'OOS base (realistic, 2025-26)':^34}")
    print(f"  {'':>4} | {'med.mo  $60->1yr  P(ruin) DD':>34} | {'med.mo  $60->1yr  P(ruin) DD':>34}")
    for k in [1.0, 1.5, 2.0, 3.0]:
        rf = tgt.block_bootstrap(full_arr, k, n_paths=3000)
        ro = tgt.block_bootstrap(oos_arr, k, n_paths=3000)
        f60 = 60 * rf["median_year_x"]; o60 = 60 * ro["median_year_x"]
        print(f"  {k:>4.1f} | {rf['median_monthly']*100:>5.1f}% ${f60:>6.0f}  {rf['p_ruin']*100:>4.0f}% "
              f"{rf['median_maxDD']*100:>4.0f}% | {ro['median_monthly']*100:>5.1f}% ${o60:>6.0f}  "
              f"{ro['p_ruin']*100:>4.0f}% {ro['median_maxDD']*100:>4.0f}%")

    # ---- save artifacts ----
    out = lab  # paths
    (1 + net8.fillna(0)).cumprod().rename("equity").to_csv("results/equity_top8.csv")
    m.rename("monthly_return").to_csv("results/monthly_top8.csv")
    # today's target weights (last row, nonzero)
    today = wv8.iloc[-1]
    today = today[today.abs() > 1e-6].sort_values(key=abs, ascending=False)
    today.rename("target_weight").to_csv("results/today_target_weights.csv")
    print("\nCurrent target weights (last bar in data, vol-targeted gross):")
    for sym, w in today.items():
        side = "LONG " if w > 0 else "SHORT"
        print(f"   {side} {sym:<10} {w*100:+6.1f}%")
    print(f"   (gross={today.abs().sum()*100:.0f}% of equity at 1x; multiply by your leverage)")
    print("\nSaved: results/equity_top8.csv, results/monthly_top8.csv, results/today_target_weights.csv")


if __name__ == "__main__":
    main()
