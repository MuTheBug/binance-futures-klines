"""Round 2: stack the robust winners (weight-smoothing + multi-signal blend) and add
range-based (Garman-Klass) vol sizing. Goal: robust Sharpe toward 2.0, IS AND OOS."""
from __future__ import annotations
import numpy as np, pandas as pd
import data, engine, strategies as st, lab

I = "1d"; TV = 0.40


def gk_sigma(window, c):
    """Garman-Klass per-bar daily vol (uses OHLC; more efficient than close-to-close)."""
    o = data.load("open", I).reindex_like(c); h = data.load("high", I).reindex_like(c)
    l = data.load("low", I).reindex_like(c); cl = data.load("close", I).reindex_like(c)
    rs = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(cl / o) ** 2
    return np.sqrt(rs.rolling(window, min_periods=window // 2).mean())


def sig_to_w(signal, sigma, elig, gross=1.0):
    raw = (signal / sigma.replace(0, np.nan)).where(elig, 0.0)
    s = raw.abs().sum(axis=1).replace(0, np.nan)
    return raw.div(s, axis=0).mul(gross).fillna(0.0)


def vt_sim(w, r, target=TV):
    return engine.simulate(engine.vol_target(w, r, I, target_vol=target), r, I)["net"]


def io(net):
    return engine.metrics(net, I), engine.metrics(lab.split(net)[0], I), engine.metrics(lab.split(net)[1], I)


def line(tag, net, base=None):
    f, mi, mo = io(net)
    extra = ""
    if base is not None:
        cc = pd.concat([net, base], axis=1).dropna(); extra = f" corr={cc.iloc[:,0].corr(cc.iloc[:,1]):.2f}"
    print(f"  {tag:<32} FULL={f['Sharpe']:>5.2f} IS={mi['Sharpe']:>5.2f} OOS={mo['Sharpe']:>5.2f} "
          f"Cal={f['Calmar']:>4.2f} DD={f['maxDD']*100:>4.0f}%{extra}")
    return net


def blend_signal(c):
    """Average of three trend-family signals, each in [-1,1]."""
    s_str = st.strength_signal(c, lookbacks=(15, 30, 60, 90), k=2.0)
    hi = c.rolling(40, min_periods=40).max(); lo = c.rolling(40, min_periods=40).min()
    s_don = pd.DataFrame(0.0, index=c.index, columns=c.columns).mask(c >= hi, 1).mask(c <= lo, -1)
    s_don = s_don.replace(0.0, np.nan).ffill().fillna(0.0)
    s_sma = sum(np.sign(c / c.rolling(L).mean() - 1) for L in (15, 30, 60, 90)) / 4
    return (s_str + s_don + s_sma) / 3


def main():
    c, v, r, e = lab.load_data(I)
    cc_sigma = r.rolling(15, min_periods=8).std()          # close-to-close sigma (baseline)
    gk = gk_sigma(15, c)

    print("="*108); print("BASELINE & EMA plateau extension"); print("="*108)
    bw = st.concentrate(st.ts_trend_strength(c, v, lookbacks=(15, 30, 60, 90), vol_lookback=15, k=2.0, elig=e), 6)
    base = line("baseline top6", vt_sim(bw, r))
    for N in [5, 8, 10, 12, 15]:
        line(f"baseline + weightEMA{N}", vt_sim(bw.ewm(span=N).mean(), r), base)

    print("\n" + "="*108); print("RANGE-VOL (Garman-Klass) sizing vs close-to-close"); print("="*108)
    str_sig = st.strength_signal(c, lookbacks=(15, 30, 60, 90), k=2.0)
    w_cc = st.concentrate(sig_to_w(str_sig, cc_sigma, e), 6)
    w_gk = st.concentrate(sig_to_w(str_sig, gk, e), 6)
    line("strength + cc-vol", vt_sim(w_cc, r), base)
    line("strength + gk-vol", vt_sim(w_gk, r), base)

    print("\n" + "="*108); print("MULTI-SIGNAL BLEND (+ gk-vol) and + EMA"); print("="*108)
    bs = blend_signal(c)
    wb_cc = st.concentrate(sig_to_w(bs, cc_sigma, e), 6)
    wb_gk = st.concentrate(sig_to_w(bs, gk, e), 6)
    line("blend + cc-vol", vt_sim(wb_cc, r), base)
    line("blend + gk-vol", vt_sim(wb_gk, r), base)
    for N in [3, 5, 8]:
        line(f"blend gk + weightEMA{N}", vt_sim(wb_gk.ewm(span=N).mean(), r), base)

    print("\n" + "="*108); print(">>> STACKED FINALS (blend + gk-vol + EMA), vary concentration K"); print("="*108)
    for K in [5, 6, 8]:
        wbk = st.concentrate(sig_to_w(bs, gk, e), K)
        for N in [5, 8]:
            line(f"K{K} blend+gk+EMA{N}", vt_sim(wbk.ewm(span=N).mean(), r), base)


if __name__ == "__main__":
    main()
