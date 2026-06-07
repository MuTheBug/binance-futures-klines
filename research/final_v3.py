"""DEFINITIVE v3 (runnable): strength top-6 select -> EMA-span-ensemble smooth -> hold top-10.
Tearsheet + leverage/ruin + $60 projection + today's weights. Saves artifacts."""
from __future__ import annotations
import numpy as np, pandas as pd
import data, engine, strategies as st, lab, target as tgt, funding

I = "1d"; TV = 0.40; K_SELECT = 6; K_HOLD = 10; SPANS = (5, 10, 15); LBS = (15, 30, 60, 90); KSTEEP = 2.0
# tokenized stocks / metals / FX — exclude so the edge is pure crypto (they only exist 2025-26)
EXCLUDE = {"NVDA","TSLA","MSTR","SPY","QQQ","INTC","MU","MRVL","AMD","SNDK","SKHYNIX","NOK",
           "EWY","SOXL","CRCL","SPCX","XAG","XAU","XAUT","PAXG","CL","BZ","SLX","BILL","COIN","GENIUS","CHIP"}


def build(c, v, e):
    w6 = st.concentrate(st.ts_trend_strength(c, v, lookbacks=LBS, vol_lookback=15, k=KSTEEP, elig=e), K_SELECT)
    return st.concentrate(st.ema_ensemble(w6, SPANS), K_HOLD)


def main():
    c, v, r, e = lab.load_data(I)
    e = e & ~e.columns.isin(EXCLUDE)           # crypto-only
    F = funding.load_daily_funding(c.index).reindex(columns=c.columns)  # real per-coin funding
    w = build(c, v, e)
    sim = engine.simulate(engine.vol_target(w, r, I, target_vol=TV, funding_matrix=F), r, I, funding_matrix=F)
    net = sim["net"]

    print("#"*94); print("DEFINITIVE v3 (runnable): risk-adj trend strength + EMA-ensemble smoothing, hold top-10"); print("#"*94)
    for tag, seg in [("FULL", net), ("IS  ", lab.split(net)[0]), ("OOS ", lab.split(net)[1])]:
        print(engine.fmt_metrics(engine.metrics(seg, I, "v3 " + tag)))
    eq = (1 + net.fillna(0)).cumprod(); yr = eq.resample("YE").last().pct_change(); yr.iloc[0] = eq.resample("YE").last().iloc[0]-1
    print("\nyearly: " + "  ".join(f"{t.year}:{x*100:+.0f}%" for t, x in yr.dropna().items()))
    ms = engine.monthly_summary(net)
    print(f"monthly: mean={ms['mean_monthly']*100:.1f}% median={ms['median_monthly']*100:.1f}% "
          f"%+={ms['pct_positive']*100:.0f}% best={ms['best']*100:.0f}% worst={ms['worst']*100:.0f}% "
          f"| turnover/day={sim['turnover'].mean():.2f}  median #positions={int((w.abs()>1e-4).sum(axis=1).replace(0,np.nan).median())}")

    print("\n" + "="*94); print("LEVERAGE vs RUIN ($60, 1yr, block-bootstrap w/ liquidation)"); print("="*94)
    is_a = lab.split(net)[0].dropna().to_numpy(); full_a = net.dropna().to_numpy(); oos_a = lab.split(net)[1].dropna().to_numpy()
    print(f"  {'lev':>4} | {'IS-base (conservative)':>22} | {'FULL-base':>16} | {'OOS-base (hot)':>16} | P(ruin)")
    for k in [1.0, 1.5, 2.0, 3.0]:
        ri = tgt.block_bootstrap(is_a, k, n_paths=3000); rf = tgt.block_bootstrap(full_a, k, n_paths=3000); ro = tgt.block_bootstrap(oos_a, k, n_paths=3000)
        print(f"  {k:>4.1f} | {ri['median_monthly']*100:>7.1f}%/mo ${60*ri['median_year_x']:>5.0f} | "
              f"{rf['median_monthly']*100:>5.1f}% ${60*rf['median_year_x']:>5.0f} | "
              f"{ro['median_monthly']*100:>5.1f}% ${60*ro['median_year_x']:>5.0f} | {rf['p_ruin']*100:.0f}%")

    today = w.iloc[-1]; today = today[today.abs() > 1e-4].sort_values(key=abs, ascending=False)
    print(f"\nToday's target ({c.index[-1].date()}), gross {today.abs().sum()*100:.0f}% @1x (×leverage), {len(today)} positions:")
    for s, x in today.items():
        print(f"   {'LONG ' if x>0 else 'SHORT'} {s:<10} {x*100:+6.1f}%")

    (1 + net.fillna(0)).cumprod().rename("equity").to_csv("results/equity_v3.csv")
    engine.monthly_returns(net).rename("monthly_return").to_csv("results/monthly_v3.csv")
    today.rename("target_weight").to_csv("results/today_target_weights.csv")
    print("\nsaved results/equity_v3.csv, monthly_v3.csv, today_target_weights.csv")


if __name__ == "__main__":
    main()
