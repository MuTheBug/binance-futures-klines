"""
Push Sharpe toward 2.0 — ROBUSTLY. Each overlay must help BOTH IS and OOS.
Levers: (1) diversification across low-corr sleeves, (2) crash/vol overlay to cut the
left tail, (3) smoothing/lag-averaging to cut turnover & timing noise, (4) more bets.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import data, engine, strategies as st, lab

I = "1d"; TV = 0.40


def base_weights(c, v, e, lookbacks=(15, 30, 60, 90), k=2.0, kcat=6, vlb=15):
    w = st.ts_trend_strength(c, v, lookbacks=lookbacks, vol_lookback=vlb, k=k, elig=e)
    return st.concentrate(w, kcat)


def vt_sim(w, r, target=TV, vol_lb=None):
    wv = engine.vol_target(w, r, I, target_vol=target, vol_lookback=vol_lb)
    return engine.simulate(wv, r, I)["net"]


def io(net):
    mi, mo = engine.metrics(lab.split(net)[0], I), engine.metrics(lab.split(net)[1], I)
    f = engine.metrics(net, I)
    return f, mi, mo


def line(tag, net, base=None):
    f, mi, mo = io(net)
    extra = ""
    if base is not None:
        cc = pd.concat([net, base], axis=1).dropna()
        extra = f"  corr2base={cc.iloc[:,0].corr(cc.iloc[:,1]):.2f}"
    print(f"  {tag:<30} FULL Sh={f['Sharpe']:>5.2f} | IS={mi['Sharpe']:>5.2f} | OOS={mo['Sharpe']:>5.2f} "
          f"| CAGR={f['CAGR']*100:>5.0f}% DD={f['maxDD']*100:>4.0f}%{extra}")
    return net


def main():
    c, v, r, e = lab.load_data(I)
    mkt = r.where(e).mean(axis=1)                      # equal-weight market return

    print("="*112); print("BASELINE (recommended top-6, risk-adj strength)"); print("="*112)
    bw = base_weights(c, v, e)
    base = line("baseline", vt_sim(bw, r))

    print("\n" + "="*112); print("(1) SMOOTHING / LAG-AVERAGING the weights (cut turnover+timing noise)"); print("="*112)
    for N in [2, 3, 5, 8]:
        line(f"weight-EMA{N}", vt_sim(bw.ewm(span=N).mean(), r), base)

    print("\n" + "="*112); print("(2) CRASH / VOL OVERLAY (de-gross when market vol spikes)"); print("="*112)
    for win, cap in [(10, 1.0), (20, 1.0), (10, 1.5)]:
        mvol = mkt.rolling(win, min_periods=win // 2).std() * np.sqrt(365)
        ov = (0.60 / mvol).shift(1).clip(0, cap).fillna(0.0)     # scale gross by market-vol target
        line(f"crash-overlay w{win} cap{cap}", vt_sim(bw.mul(ov, axis=0), r), base)

    print("\n" + "="*112); print("(3) MULTI-SPEED TREND SLEEVES (fast/med/slow), each vol-targeted then averaged"); print("="*112)
    fast = vt_sim(base_weights(c, v, e, lookbacks=(7, 14, 21)), r)
    med = vt_sim(base_weights(c, v, e, lookbacks=(15, 30, 60, 90)), r)
    slow = vt_sim(base_weights(c, v, e, lookbacks=(60, 120, 180)), r)
    line("fast(7-21)", fast, base); line("med(15-90)", med, base); line("slow(60-180)", slow, base)
    fs = pd.concat([fast, med, slow], axis=1)
    line("avg fast+med+slow", fs.mean(axis=1), base)

    print("\n" + "="*112); print("(4) REVERSAL DIVERSIFIER (market-neutral, fade recent movers) + combine"); print("="*112)
    for lb in [2, 3, 5]:
        rev_w = st.xs_reversal(c, v, lookback=lb, vol_lookback=15, elig=e, quantile=0.25)
        rev_w = rev_w.ewm(span=5).mean()                          # smooth to cut turnover
        rev = vt_sim(rev_w, r)
        line(f"reversal lb{lb} (alone)", rev, base)
    # combine best-guess reversal with trend
    rev_w = st.xs_reversal(c, v, lookback=3, vol_lookback=15, elig=e, quantile=0.25).ewm(span=5).mean()
    rev = vt_sim(rev_w, r)
    for wt in [0.2, 0.35, 0.5]:
        combo = (1 - wt) * base + wt * rev
        line(f"trend+{int(wt*100)}%reversal", combo, base)

    print("\n" + "="*112); print("(5) MULTI-SIGNAL BLEND (trend-strength + donchian + price/SMA), averaged signals"); print("="*112)
    s_str = st.strength_signal(c, lookbacks=(15, 30, 60, 90), k=2.0)
    don = st.donchian(c, v, lookback=40, vol_lookback=15, elig=e)   # already weights; get sign
    s_don = np.sign(don).reindex_like(c).fillna(0.0)
    s_sma = sum(np.sign(c / c.rolling(L).mean() - 1) for L in (15, 30, 60, 90)) / 4
    blend = (s_str + s_don + s_sma) / 3
    bw_blend = st.concentrate(st.signal_to_weights(blend.reindex_like(c).fillna(0), r, e, 15), 6)
    line("trend+donchian+sma blend", vt_sim(bw_blend, r), base)

    print("\n" + "="*112); print("(6) STACKED: multi-speed avg + weight-EMA + crash overlay"); print("="*112)
    # build multi-speed at weight level, smooth, overlay
    wf = base_weights(c, v, e, lookbacks=(7, 14, 21))
    wm = base_weights(c, v, e, lookbacks=(15, 30, 60, 90))
    ws = base_weights(c, v, e, lookbacks=(60, 120, 180))
    wcomb = (wf + wm + ws) / 3
    wcomb_s = wcomb.ewm(span=3).mean()
    mvol = mkt.rolling(10, min_periods=5).std() * np.sqrt(365)
    ov = (0.60 / mvol).shift(1).clip(0, 1.0).fillna(0.0)
    line("multispeed+EMA3", vt_sim(wcomb_s, r), base)
    line("multispeed+EMA3+crash", vt_sim(wcomb_s.mul(ov, axis=0), r), base)


if __name__ == "__main__":
    main()
