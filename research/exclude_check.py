import numpy as np, pandas as pd, engine, strategies as st, lab
from final_v3 import build, I, TV
c,v,r,e = lab.load_data(I)
# tokenized stocks / metals / FX (not crypto)
EXC = {"NVDA","TSLA","MSTR","SPY","QQQ","INTC","MU","MRVL","AMD","SNDK","SKHYNIX","NOK",
       "EWY","SOXL","CRCL","SPCX","XAG","XAU","XAUT","PAXG","CL","BZ","SLX","BILL","COIN",
       "AMD","GENIUS","CHIP","BTW","ZEST","SNDK"}
present = [s for s in EXC if s in c.columns]
print("excluded non-crypto present in universe:", sorted(present))
def runv3(elig):
    w6 = st.concentrate(st.ts_trend_strength(c,v,lookbacks=(15,30,60,90),vol_lookback=15,k=2.0,elig=elig),6)
    w  = st.concentrate(st.ema_ensemble(w6,(5,10,15)),10)
    net= engine.simulate(engine.vol_target(w,r,I,target_vol=TV),r,I)["net"]
    f,mi,mo=engine.metrics(net,I),engine.metrics(lab.split(net)[0],I),engine.metrics(lab.split(net)[1],I)
    return f,mi,mo,net
for label,elig in [("ALL (incl. stocks)",e),("CRYPTO-ONLY",e & ~e.columns.isin(EXC))]:
    f,mi,mo,net=runv3(elig)
    print(f"  {label:<20} FULL Sh={f['Sharpe']:.2f} CAGR={f['CAGR']*100:.0f}% | IS Sh={mi['Sharpe']:.2f} | OOS Sh={mo['Sharpe']:.2f} CAGR={mo['CAGR']*100:.0f}% DD={f['maxDD']*100:.0f}%")
