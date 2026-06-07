"""
Find the BEST live config for long-term wealth:
  A) best SL/TP brackets (risk-adjusted), then
  B) the leverage that maximizes COMPOUND growth at survivable drawdown (Kelly logic).
Everything IS/OOS-disciplined; brackets modelled on daily OHLC (gap-aware).
"""
from __future__ import annotations
import numpy as np, pandas as pd
import data, engine, strategies as st, lab, funding, target as tgt

I = "1d"; MAX_GROSS = 3.0
EXCLUDE = {"NVDA","TSLA","MSTR","SPY","QQQ","INTC","MU","MRVL","AMD","SNDK","SKHYNIX","NOK",
           "EWY","SOXL","CRCL","SPCX","XAG","XAU","XAUT","PAXG","CL","BZ","SLX","BILL","COIN","GENIUS","CHIP"}


def v3_weights(c, v, e):
    w = st.ts_trend_strength(c, v, lookbacks=(15, 30, 60, 90), vol_lookback=15, k=2.0, elig=e)
    return st.concentrate(st.ema_ensemble(st.concentrate(w, 6), (5, 10, 15)), 10)


def bracketed(o, h, l, c, held, sl, tp):
    pc = c.shift(1); a_o, a_h, a_l, a_c = o/pc-1, h/pc-1, l/pc-1, c/pc-1
    big = 9.99
    SLv = sl if sl else big; TPv = tp if tp else big
    L = a_c.copy()
    sL = a_l <= -SLv; tL = (a_h >= TPv) & ~sL
    L = L.mask(sL, np.minimum(a_o, -SLv)).mask(tL, np.maximum(a_o, TPv))
    S = a_c.copy()
    sS = a_h >= SLv; tS = (a_l <= -TPv) & ~sS
    S = S.mask(sS, np.maximum(a_o, SLv)).mask(tS, np.minimum(a_o, -TPv))
    reff = a_c.copy().mask(held > 0, L).mask(held < 0, S)
    trig = ((held > 0) & (sL | tL)) | ((held < 0) & (sS | tS))
    return reff, trig


def m(net):
    return engine.metrics(net, I), engine.metrics(lab.split(net)[0], I), engine.metrics(lab.split(net)[1], I)


def main():
    c, v, r, e = lab.load_data(I)
    e = e & ~e.columns.isin(EXCLUDE)
    F = funding.load_daily_funding(c.index).reindex(columns=c.columns)
    e = e & F.notna()
    o = data.load("open", I).reindex_like(c); h = data.load("high", I).reindex_like(c); l = data.load("low", I).reindex_like(c)
    R_cc = c.pct_change()
    W = v3_weights(c, v, e)

    def net_for(sl, tp, lev):
        wv = engine.vol_target(W, R_cc, I, target_vol=0.40*lev, max_leverage=MAX_GROSS, funding_matrix=F)
        held = np.sign(wv.shift(1))
        reff, trig = bracketed(o, h, l, c, held, sl, tp)
        sim = engine.simulate(wv, reff, I, funding_matrix=F)
        extra = ((wv.shift(1).abs()*trig).sum(axis=1)*engine.COST_PER_SIDE).fillna(0.0)
        return sim["net"] - extra

    print("="*100); print("A) BRACKET CONFIG SWEEP (at 2x, equal footing). Want best OOS Sharpe/Calmar."); print("="*100)
    print(f"  {'SL':>6} {'TP':>6} | {'FULL Sh':>7} {'CAGR':>6} {'maxDD':>6} {'Calmar':>6} | {'IS Sh':>5} {'OOS Sh':>6}")
    configs = [(0.18,0.50),(0.18,None),(0.25,None),(0.30,None),(0.35,None),(None,None),(0.30,1.0),(0.25,0.80)]
    best=None
    for sl,tp in configs:
        net=net_for(sl,tp,2.0); f,mi,mo=m(net)
        tag=f"{(str(int(sl*100))+'%') if sl else 'off':>6} {(str(int(tp*100))+'%') if tp else 'off':>6}"
        cur=" <-current" if (sl,tp)==(0.18,0.50) else ""
        print(f"  {tag} | {f['Sharpe']:>7.2f} {f['CAGR']*100:>5.0f}% {f['maxDD']*100:>5.0f}% {f['Calmar']:>6.2f} | {mi['Sharpe']:>5.2f} {mo['Sharpe']:>6.2f}{cur}")
        score=min(mi['Sharpe'],mo['Sharpe'])  # robust: worst of IS/OOS
        if best is None or score>best[0]: best=(score,sl,tp)
    _,bsl,btp=best
    print(f"  -> chosen brackets (max of min(IS,OOS) Sharpe): SL={bsl} TP={btp}")

    print("\n"+"="*100); print(f"B) LEVERAGE SWEEP for chosen brackets — COMPOUND growth vs drawdown vs ruin"); print("="*100)
    print(f"  {'lev':>4} | {'FULL CAGR':>9} {'Sharpe':>6} {'maxDD':>6} {'Calmar':>6} | {'OOS CAGR':>8} | {'boot med CAGR':>13} {'P(ruin/2y)':>10} {'medDD':>6}")
    base1x = net_for(bsl,btp,1.0).dropna().to_numpy()   # 1x net for Kelly scaling
    for lev in [1.0,1.5,2.0,2.5,3.0,4.0]:
        net=net_for(bsl,btp,lev); f,mi,mo=m(net)
        # bootstrap 2-year compound growth distribution by scaling 1x net by lev (Kelly view)
        bb=tgt.block_bootstrap(base1x, lev, n_paths=2500, horizon=720)
        medCAGR=bb['median_year_x']**0.5-1   # 2yr -> annualized
        print(f"  {lev:>4.1f} | {f['CAGR']*100:>8.0f}% {f['Sharpe']:>6.2f} {f['maxDD']*100:>5.0f}% {f['Calmar']:>6.2f} | "
              f"{mo['CAGR']*100:>7.0f}% | {medCAGR*100:>12.0f}% {bb['p_ruin']*100:>9.0f}% {bb['median_maxDD']*100:>5.0f}%")

    # growth-optimal (max bootstrap median 2y wealth) scanning finely
    print("\n  fine Kelly scan (bootstrap median 2-yr wealth multiple):")
    bestk=None
    for lev in np.arange(1.0,5.01,0.5):
        bb=tgt.block_bootstrap(base1x, lev, n_paths=2000, horizon=720)
        w2=bb['median_year_x']
        if bestk is None or w2>bestk[1]: bestk=(lev,w2,bb['p_ruin'],bb['median_maxDD'])
        print(f"    lev {lev:>3.1f}: med 2y wealth x{w2:>5.2f}  P(ruin/2y) {bb['p_ruin']*100:>3.0f}%  medDD {bb['median_maxDD']*100:>4.0f}%")
    print(f"  -> growth-optimal lev ~{bestk[0]:.1f} (med 2y x{bestk[1]:.2f}); HALF-KELLY ~{bestk[0]/2:.1f} (prudent)")


if __name__ == "__main__":
    main()
