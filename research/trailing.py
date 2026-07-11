"""Profit-trailing overlays on the APEX book -- IS-gated like everything else.

Variant P: per-position trailing stop. While a name is held (weight sign
   unchanged), track its cumulative return; if it retraces `trail` from its
   running peak, zero the weight until the book itself drops/flips the name.
Variant Q: portfolio trailing. When strategy equity retraces `dd_trig` from
   its high-water mark, scale the whole book by `throttle` until drawdown
   recovers to dd_trig/2. Costs of throttling are fully charged.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import altdata
import echo
import engine
import lab
import strategies as st
from apex import hold_cadence

close, vol, ret, elig = lab.load_data("1d")


def build_book():
    sigT = st.strength_signal(close, lookbacks=(15, 30, 60, 90),
                              strength_vol=30, k=2.0)
    A = st.concentrate(st.ema_ensemble(
        st.signal_to_weights(sigT, ret, elig, 15), (5, 10, 15)), 10)
    B = echo.residual_continuation(close, ret, elig, beta_window=60,
                                   formations=(3, 5, 8), cap=0.10)
    ls = altdata.load_metric("toptrader_pos_ls", reindex_like=close)
    C = echo.positioning_sleeve(close, ret, elig, ls, fade=True, lag=1,
                                beta_window=60, cap=0.10)
    return hold_cadence(0.40 * A + 0.40 * B + 0.20 * C, 3)


def per_position_trailing(W, trail):
    """Zero a held name after it retraces `trail` from its peak cum-return
    (long: price peak; short: mirror). Kill flag persists until the book
    itself drops or flips the name. Uses returns through day t to modify
    W(t), which the engine lags one bar -> no look-ahead."""
    Wv = W.values
    R = ret.fillna(0.0).values
    n_bars, n_sym = Wv.shape
    out = Wv.copy()
    sign = np.zeros(n_sym)
    cum = np.zeros(n_sym)
    peak = np.zeros(n_sym)
    killed = np.zeros(n_sym, dtype=bool)
    for t in range(n_bars):
        s_now = np.sign(Wv[t])
        new_ep = s_now != sign
        sign = s_now
        cum[new_ep] = 0.0
        peak[new_ep] = 0.0
        killed[new_ep] = False
        # update episode PnL with today's return (known at close t)
        pnl = sign * R[t]
        cum += pnl
        np.maximum(peak, cum, out=peak)
        killed |= (peak - cum) > trail
        out[t] = np.where(killed, 0.0, Wv[t])
    return pd.DataFrame(out, index=W.index, columns=W.columns)


def portfolio_trailing(W, base_net, dd_trig, throttle):
    eq = (1 + base_net.fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    f = np.ones(len(dd))
    on = False
    ddv = dd.values
    for t in range(len(ddv)):
        if not on and ddv[t] <= -dd_trig:
            on = True
        elif on and ddv[t] >= -dd_trig / 2:
            on = False
        f[t] = throttle if on else 1.0
    fac = pd.Series(f, index=W.index).shift(1).fillna(1.0)
    return W.mul(fac, axis=0)


def net_of(w):
    wv = engine.vol_target(w, ret, "1d", target_vol=0.40, max_leverage=3.0)
    return engine.simulate(wv, ret, "1d")["net"]


def row(net, label):
    i, o = lab.split(net)
    mi = engine.metrics(i, "1d")
    mf = engine.metrics(net, "1d")
    return {"variant": label, "IS_sharpe": mi["Sharpe"], "IS_maxDD": mi["maxDD"],
            "IS_calmar": mi["Calmar"], "full_sharpe": mf["Sharpe"],
            "full_CAGR": mf["CAGR"], "full_maxDD": mf["maxDD"],
            "OOS_sharpe": engine.metrics(o, "1d")["Sharpe"]}


def main():
    W = build_book()
    base_net = net_of(W)
    rows = [row(base_net, "APEX base (no trailing)")]

    for trail in (0.10, 0.20, 0.30):
        rows.append(row(net_of(per_position_trailing(W, trail)),
                        f"P: per-position trail {trail:.0%}"))
    for dd_trig, thr in ((0.15, 0.5), (0.25, 0.5), (0.15, 0.0), (0.25, 0.0)):
        rows.append(row(net_of(portfolio_trailing(W, base_net, dd_trig, thr)),
                        f"Q: portfolio dd>{dd_trig:.0%} -> x{thr}"))

    df = pd.DataFrame(rows)
    df.to_csv("results/trailing_results.csv", index=False)
    print(df.round(3).to_string(index=False))
    print("\nNOTE: IS columns decide; OOS shown for the record (single read, "
          "all variants reported, none cherry-picked).")


if __name__ == "__main__":
    main()
