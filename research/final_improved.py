"""Definitive recommended config: risk-adjusted trend strength, select-broad/hold top-6,
vol-targeted. Prints tearsheet + leverage table + today's weights; saves artifacts."""
from __future__ import annotations
import numpy as np, pandas as pd
import data, engine, strategies as st, lab, target as tgt

INTERVAL="1d"; TV=0.40; KCAT=6; K_STEEP=2.0; LBS=(15,30,60,90)

def main():
    c,v,r,e = lab.load_data(INTERVAL)
    w = st.ts_trend_strength(c,v,lookbacks=LBS,vol_lookback=15,k=K_STEEP,elig=e)
    w = st.concentrate(w,KCAT)
    w = engine.vol_target(w,r,INTERVAL,target_vol=TV)
    net = engine.simulate(w,r,INTERVAL)["net"]

    print("#"*92); print("RECOMMENDED: risk-adjusted trend strength | select broad, hold top-6 | vol-target 40%"); print("#"*92)
    for tag,seg in [("FULL",net),("IS  ",lab.split(net)[0]),("OOS ",lab.split(net)[1])]:
        print(engine.fmt_metrics(engine.metrics(seg,INTERVAL,"top6 "+tag)))
    eq=(1+net.fillna(0)).cumprod(); yr=eq.resample("YE").last().pct_change(); yr.iloc[0]=eq.resample("YE").last().iloc[0]-1
    print("\nyearly: "+"  ".join(f"{t.year}:{x*100:+.0f}%" for t,x in yr.dropna().items()))
    ms=engine.monthly_summary(net)
    print(f"monthly: mean={ms['mean_monthly']*100:.1f}% median={ms['median_monthly']*100:.1f}% "
          f"%+={ms['pct_positive']*100:.0f}% best={ms['best']*100:.0f}% worst={ms['worst']*100:.0f}%")
    m=engine.monthly_returns(net)
    print("last 12 months: "+"  ".join(f"{t.strftime('%y-%m')}:{x*100:+.0f}%" for t,x in m.tail(12).items()))

    print("\n"+"="*92); print("LEVERAGE vs RUIN — $60 over 1 year (block-bootstrap w/ liquidation)"); print("="*92)
    full_arr=net.dropna().to_numpy(); oos_arr=lab.split(net)[1].dropna().to_numpy()
    print(f"  {'lev':>4} |   FULL: med/mo  $60->1yr  P(ruin) |    OOS: med/mo  $60->1yr  P(ruin)")
    for k in [1.0,1.5,2.0,3.0]:
        rf=tgt.block_bootstrap(full_arr,k,n_paths=3000); ro=tgt.block_bootstrap(oos_arr,k,n_paths=3000)
        print(f"  {k:>4.1f} |    {rf['median_monthly']*100:>6.1f}%   ${60*rf['median_year_x']:>5.0f}    {rf['p_ruin']*100:>4.0f}%  "
              f"|    {ro['median_monthly']*100:>6.1f}%   ${60*ro['median_year_x']:>5.0f}    {ro['p_ruin']*100:>4.0f}%")

    today=w.iloc[-1]; today=today[today.abs()>1e-6].sort_values(key=abs,ascending=False)
    print(f"\nToday's target ({c.index[-1].date()}), gross {today.abs().sum()*100:.0f}% at 1x (×your leverage):")
    for s,x in today.items(): print(f"   {'LONG ' if x>0 else 'SHORT'} {s:<10} {x*100:+6.1f}%")

    (1+net.fillna(0)).cumprod().rename("equity").to_csv("results/equity_recommended.csv")
    m.rename("monthly_return").to_csv("results/monthly_recommended.csv")
    today.rename("target_weight").to_csv("results/today_target_weights.csv")
    print("\nsaved: results/equity_recommended.csv, monthly_recommended.csv, today_target_weights.csv")

if __name__=="__main__": main()
