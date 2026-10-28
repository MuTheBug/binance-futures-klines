"""Swing-trading system: discrete trades with entries, ATR stops and exits (days to weeks).

Pre-committed protocol (written BEFORE any run):
  * Coins are split at random (seed 7) into DEV and HOLDOUT halves. Rules are designed on
    DEV coins, data <= 2024-12-31 only.
  * Test A (clean): HOLDOUT coins, <= 2024-12-31  -- never looked at during design.
  * Test B: all coins, 2025-01 -> 2026-06 (this window was seen by earlier repo research, but
    not for these rules).
  * Success = Sharpe >= 1.5 on BOTH tests and >= 2.0 on the full sample. Anything else is
    reported as a miss.

Mechanics (no look-ahead): signals use the daily close of bar t; entry at the close of t
(1-bar-delayed variant tested in robustness). Stops are checked against bar t+1's high/low.
A stop fills at the stop price, or at the open if the bar gaps through it (pessimistic).
Costs: 7bps/side plus an extra 10bps of slippage on stop fills. Funding: longs pay 11%/yr,
shorts receive nothing.
Positions are sized so that hitting the initial stop costs RISK of equity, with a per-name
notional cap.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import data, engine, strategies as st, lab
from final_v3 import EXCLUDE

I = "1d"
COST, STOP_SLIP, FUND_LONG = 0.0007, 0.0010, 0.11 / 365
P = dict(don=20, atr=14, stop_atr=2.0, trail_atr=3.0, max_days=30, trend_min=0.0,
         risk=0.005, cap=0.20, vol_mult=0.0, sides=(1, -1), delay=0,
         setup="breakout", regime=False, ignite_thr=0.25, hedge=False, pull_atr=1.5)


def load():
    o, h, l, c, v = (data.load(f, I) for f in ("open", "high", "low", "close", "volume"))
    keep = [x for x in c.columns if x not in EXCLUDE]
    sl = lambda d: d.loc[d.index >= "2020-05-31", keep]
    return sl(o), sl(h), sl(l), sl(c), sl(v)


def split_coins(cols, seed=7):
    rng = np.random.default_rng(seed)
    cols = sorted(cols); idx = rng.permutation(len(cols))
    half = len(cols) // 2
    return [cols[i] for i in idx[:half]], [cols[i] for i in idx[half:]]


def atr(h, l, c, n):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()]).groupby(level=0).max()
    return tr.reindex(c.index).rolling(n, min_periods=n).mean()


def signals(o, h, l, c, v, p, e=None):
    e = st.eligibility(c, v, 60, 5e6, 30) if e is None else e
    trend = st.strength_signal(c)                                    # v3's risk-adj trend in [-1,1]
    hi = h.rolling(p["don"], min_periods=p["don"]).max().shift(1)   # prior N-day channel
    lo = l.rolling(p["don"], min_periods=p["don"]).min().shift(1)
    volok = v > p["vol_mult"] * v.rolling(20, min_periods=10).mean().shift(1)
    if p["setup"] == "ignite":                                       # one-day idiosyncratic pump
        r = c.pct_change()
        long_e = (r.sub(r["BTC"], axis=0) > p["ignite_thr"]) & e
        short_e = long_e & False
    elif p["setup"] == "pullback":                                   # buy the dip inside a strong trend
        drop = (c / c.shift(3) - 1) / (A_ := atr(h, l, c, p["atr"]) / c)
        long_e = (trend > p["trend_min"]) & (drop < -p["pull_atr"]) & e
        short_e = (trend < -p["trend_min"]) & (drop > p["pull_atr"]) & e
    else:
        long_e = (c > hi) & (trend > p["trend_min"]) & e & volok
        short_e = (c < lo) & (trend < -p["trend_min"]) & e & volok
    if p["regime"]:                                                  # BTC must be trending the same way
        bt = trend["BTC"]
        long_e = long_e & (bt > 0).to_numpy()[:, None]
        short_e = short_e & (bt < 0).to_numpy()[:, None]
    return long_e, short_e, atr(h, l, c, p["atr"])


def backtest(o, h, l, c, v, p=P, cols=None, collect=False):
    """Returns daily portfolio return series (+ trade list if collect)."""
    cols = cols or list(c.columns)
    cost = COST
    if p["hedge"]:      # trade the coin/BTC ratio: every trade is coin vs an equal-notional BTC leg
        e = st.eligibility(c, v, 60, 5e6, 30)
        c = c.div(c["BTC"], axis=0)
        o = h = l = c                                  # close-only ratio bars: stops act on the close
        cols = [x for x in cols if x != "BTC"]
        cost = 2 * COST                                # two legs
        le, se, A = signals(o, h, l, c, v, p, e)
    else:
        le, se, A = signals(o, h, l, c, v, p)
    O, H, L, C = (x[cols].to_numpy(float) for x in (o, h, l, c))
    LE, SE, AT = (x[cols].to_numpy() for x in (le, se, A))
    T, N = C.shape
    pnl = np.zeros(T)
    trades = []
    d = p["delay"]
    for j in range(N):
        pos = 0; w = 0.0; entry = stop = ext = 0.0; t0 = 0
        for t in range(1, T):
            if pos != 0:                                   # manage open trade over bar t
                prev = C[t - 1, j]
                if np.isnan(C[t, j]):                      # delisted / missing: exit at last price
                    trades.append((cols[j], pos, t0, t, entry, prev)); pos = 0; continue
                hit = (L[t, j] <= stop) if pos > 0 else (H[t, j] >= stop)
                if hit:
                    gap = (O[t, j] < stop) if pos > 0 else (O[t, j] > stop)
                    px = O[t, j] if gap else stop
                    pnl[t] += w * pos * (px / prev - 1) - w * (cost + STOP_SLIP)
                    trades.append((cols[j], pos, t0, t, entry, px)); pos = 0
                else:
                    pnl[t] += w * pos * (C[t, j] / prev - 1) - (w * FUND_LONG if pos > 0 else 0.0)
                    ext = max(ext, H[t, j]) if pos > 0 else min(ext, L[t, j])
                    a = AT[t, j]
                    if not np.isnan(a):                    # chandelier trail (only tightens)
                        stop = max(stop, ext - p["trail_atr"] * a) if pos > 0 else min(stop, ext + p["trail_atr"] * a)
                    if t - t0 >= p["max_days"]:            # time stop at close
                        pnl[t] -= w * cost
                        trades.append((cols[j], pos, t0, t, entry, C[t, j])); pos = 0
                    continue
            s = t - d                                      # signal bar (delay variant)
            if s < 0 or np.isnan(AT[s, j]) or np.isnan(C[t, j]):
                continue
            side = 1 if (LE[s, j] and 1 in p["sides"]) else (-1 if (SE[s, j] and -1 in p["sides"]) else 0)
            if side:
                entry = C[t, j]; dist = p["stop_atr"] * AT[s, j] / entry
                w = min(p["risk"] / dist, p["cap"])
                stop = entry * (1 - side * dist); ext = entry
                pos, t0 = side, t
                pnl[t] -= w * cost
    net = pd.Series(pnl, index=c.index)
    if collect:
        tr = pd.DataFrame(trades, columns=["sym", "side", "t0", "t1", "entry", "exit"])
        tr["ret"] = tr.side * (tr.exit / tr.entry - 1)
        tr["days"] = tr.t1 - tr.t0
        tr["start"] = c.index[tr.t0]
        return net, tr
    return net


def vol_target(n, tv=0.40, lb=30, cap=3.0):
    """Portfolio-level overlay: scale by lagged realized vol (no look-ahead)."""
    rv = n.rolling(lb, min_periods=lb // 2).std() * np.sqrt(365)
    return n * (tv / rv).shift(1).clip(upper=cap).fillna(0.0)


def sh(x):
    return engine.metrics(x, I)["Sharpe"] if len(x.dropna()) else float("nan")


def summary(name, net, tr=None):
    a, b = lab.split(net)
    print(f"{name:<34} FULL {sh(net):5.2f} | IS {sh(a):5.2f} | OOS {sh(b):5.2f} | "
          f"maxDD {engine.metrics(net, I)['maxDD']:6.1%} | CAGR {engine.metrics(net, I)['CAGR']:6.1%}"
          + (f" | trades {len(tr)} win {(tr.ret>0).mean():.0%} avgdays {tr.days.mean():.0f}" if tr is not None else ""))


if __name__ == "__main__":
    o, h, l, c, v = load()
    dev, hold = split_coins(c.columns)
    isonly = lambda x: x[x.index <= lab.IS_END]
    print(f"DEV {len(dev)} coins, HOLDOUT {len(hold)} coins\n-- design runs: DEV coins, IS only --")
    grid = [{}] if len(sys.argv) < 2 else [dict(don=d, trail_atr=ta, trend_min=tm)
                                             for d in (10, 20, 40) for ta in (2.0, 3.0, 4.0) for tm in (0.0, 0.3)]
    for g in grid:
        p = {**P, **g}
        n = isonly(backtest(o, h, l, c, v, p, dev))
        print(f"  {str(g):<52} IS-dev Sharpe {sh(n):5.2f}  CAGR {engine.metrics(n, I)['CAGR']:6.1%}")
