"""
Backtest the LIVE bot's exact current settings:
  * v3 signal: risk-adj trend strength (k=2, LB 15/30/60/90), top-6 -> EMA(5,10,15) -> hold top-10
  * vol-target = TARGET_VOL(0.40) * LEVERAGE(2.0) = 0.80 annualized, capped MAX_GROSS=3.0x
  * per-position SL 18% / TP 50% brackets, reset daily (modelled on daily OHLC, gap-aware)
  * crypto-only universe (tokenized stocks excluded); realistic costs + real per-coin funding
We compare WITH vs WITHOUT the SL/TP brackets to isolate their effect.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import data, engine, strategies as st, lab, funding, target as tgt

I = "1d"
TARGET_VOL_EFF = 0.40 * 2.0       # 0.40 base * LEVERAGE 2.0  (matches config.env)
MAX_GROSS = 3.0
SL, TP = 0.18, 0.50
EXCLUDE = {"NVDA","TSLA","MSTR","SPY","QQQ","INTC","MU","MRVL","AMD","SNDK","SKHYNIX","NOK",
           "EWY","SOXL","CRCL","SPCX","XAG","XAU","XAUT","PAXG","CL","BZ","SLX","BILL","COIN","GENIUS","CHIP"}


def v3_weights(c, v, e):
    w = st.ts_trend_strength(c, v, lookbacks=(15, 30, 60, 90), vol_lookback=15, k=2.0, elig=e)
    return st.concentrate(st.ema_ensemble(st.concentrate(w, 6), (5, 10, 15)), 10)


def bracketed_returns(o, h, l, c, held_sign):
    """Per-asset daily return with SL/TP applied intraday (entry = prior close), gap-aware.
    held_sign: sign of the position held over the day (from lagged weights)."""
    pc = c.shift(1)
    a_o, a_h, a_l, a_c = o/pc - 1, h/pc - 1, l/pc - 1, c/pc - 1
    # long: stop if low<=-SL (fill min(open,-SL)); else tp if high>=TP (fill max(open,TP))
    L = a_c.copy()
    sL = a_l <= -SL
    tL = (a_h >= TP) & ~sL
    L = L.mask(sL, np.minimum(a_o, -SL)).mask(tL, np.maximum(a_o, TP))
    # short: asset return the short realizes; stop if high>=SL (fill max(open,SL)); tp if low<=-TP
    S = a_c.copy()
    sS = a_h >= SL
    tS = (a_l <= -TP) & ~sS
    S = S.mask(sS, np.maximum(a_o, SL)).mask(tS, np.minimum(a_o, -TP))
    reff = a_c.copy()
    reff = reff.mask(held_sign > 0, L).mask(held_sign < 0, S)
    trig = ((held_sign > 0) & (sL | tL)) | ((held_sign < 0) & (sS | tS))   # bracket fired
    return reff, trig


def report(name, net):
    f = engine.metrics(net, I, name + " FULL")
    mi, mo = engine.metrics(lab.split(net)[0], I), engine.metrics(lab.split(net)[1], I)
    print(engine.fmt_metrics(f))
    print(f"{'':>24}  IS Sh={mi['Sharpe']:.2f} CAGR={mi['CAGR']*100:.0f}% | OOS Sh={mo['Sharpe']:.2f} CAGR={mo['CAGR']*100:.0f}%")
    ms = engine.monthly_summary(net)
    print(f"{'':>24}  monthly mean={ms['mean_monthly']*100:.1f}% median={ms['median_monthly']*100:.1f}% "
          f"+{ms['pct_positive']*100:.0f}% worst={ms['worst']*100:.0f}% best={ms['best']*100:.0f}%")
    return f


def main():
    c, v, r, e = lab.load_data(I)
    e = e & ~e.columns.isin(EXCLUDE)
    F = funding.load_daily_funding(c.index).reindex(columns=c.columns)
    e = e & F.notna()
    o = data.load("open", I).reindex_like(c); h = data.load("high", I).reindex_like(c); l = data.load("low", I).reindex_like(c)

    W = v3_weights(c, v, e)
    R_cc = c.pct_change()
    W_vt = engine.vol_target(W, R_cc, I, target_vol=TARGET_VOL_EFF, max_leverage=MAX_GROSS, funding_matrix=F)

    print("="*96)
    print(f"LIVE-SETTINGS BACKTEST  (vol-target {TARGET_VOL_EFF:.0%} = 0.40x2.0, cap {MAX_GROSS:g}x, SL {SL:.0%}/TP {TP:.0%})")
    print(f"avg gross exposure: {W_vt.abs().sum(axis=1).iloc[260:].mean():.2f}x  | universe(crypto) cols: {e.iloc[-1].sum()}")
    print("="*96)

    # WITHOUT brackets (pure daily close-to-close at 2x)
    net_nb = engine.simulate(W_vt, R_cc, I, funding_matrix=F)["net"]
    f_nb = report("2x, NO brackets", net_nb)

    # WITH SL/TP brackets
    held = np.sign(W_vt.shift(1))
    R_eff, trig = bracketed_returns(o, h, l, c, held)
    sim = engine.simulate(W_vt, R_eff, I, funding_matrix=F)
    extra_cost = ((W_vt.shift(1).abs() * trig).sum(axis=1) * engine.COST_PER_SIDE).fillna(0.0)
    net_br = sim["net"] - extra_cost
    print()
    f_br = report("2x, WITH SL18/TP50", net_br)
    trig_days = trig.any(axis=1)
    print(f"{'':>24}  bracket-trigger days: {trig_days.sum()} ({trig_days.mean()*100:.1f}% of days), "
          f"avg {trig.sum(axis=1)[trig_days].mean():.1f} positions/trigger-day")

    # $60 projection (median via block bootstrap on the WITH-brackets daily series)
    print("\n" + "="*96); print("$60 PROJECTION (current settings = 2x + brackets), block-bootstrap 1yr"); print("="*96)
    full = net_br.dropna().to_numpy(); oos = lab.split(net_br)[1].dropna().to_numpy()
    for label, arr in [("FULL", full), ("OOS ", oos)]:
        rr = tgt.block_bootstrap(arr, 1.0, n_paths=3000)   # leverage already baked in -> k=1
        print(f"  base={label}: median {rr['median_monthly']*100:+.1f}%/mo  ${60*rr['median_year_x']:.0f}/yr  "
              f"P(ruin/yr) {rr['p_ruin']*100:.0f}%  median maxDD {rr['median_maxDD']*100:.0f}%")

    # yearly (with brackets)
    eq = (1 + net_br.fillna(0)).cumprod(); yr = eq.resample("YE").last().pct_change(); yr.iloc[0] = eq.resample("YE").last().iloc[0]-1
    print("\nyearly (2x + brackets): " + "  ".join(f"{t.year}:{x*100:+.0f}%" for t, x in yr.dropna().items()))


if __name__ == "__main__":
    main()
