"""What does chasing "$100 -> $150 in a month" look like? (asked 2026-11-11)

No strategy in this repo has an edge anywhere near +50%/month, so the only way to make that the
typical outcome is leverage. This measures the price, on Binance's complete history
(2020-06 -> 2026-11-09, every 30-day window):

  A. v4 (the best strategy found) at leverage k, with the backtest edge taken at face value
  B. the same with the edge removed (same day-to-day swings, zero average): what you get if the
     backtest edge is not real. On five fresh months v4 made nothing, so B is not far-fetched.
  C. a plain leveraged BTC position, long or short at random (a pure bet on direction)

Each window is played the kindest way for the target: stop for the month as soon as the account
is up 50%; also stop when it is down 50%. "Wiped" = down 95% or more.
"""
import numpy as np
import pandas as pd

import v5_lib as L
import vision_panel as vp

DAYS = 30


def play_returns(x: np.ndarray, k: float):
    """x: daily returns at 1x. Daily-rebalanced leverage k. Returns arrays (hit_up, hit_down, end)."""
    n = len(x) - DAYS + 1
    up = np.zeros(n, bool); dn = np.zeros(n, bool); end = np.zeros(n)
    for i in range(n):
        eq = 1.0
        for r in x[i:i + DAYS]:
            eq *= max(1.0 + k * r, 0.0)
            if eq >= 1.5:
                up[i] = True; break
            if eq <= 0.5:
                dn[i] = True; break
        end[i] = eq
    return up, dn, end


def play_btc(o, h, l, c, k: float, side: int):
    """Fixed position of k x equity opened at a close, held up to 30 days. Intraday highs/lows
    decide which barrier is touched; if both on one day, the loss is assumed first."""
    n = len(c) - DAYS
    up = np.zeros(n, bool); dn = np.zeros(n, bool); end = np.zeros(n)
    for i in range(n):
        p0 = c[i]
        eq = 1.0
        for t in range(i + 1, i + 1 + DAYS):
            worst = 1 + k * side * ((l[t] if side > 0 else h[t]) / p0 - 1)
            best = 1 + k * side * ((h[t] if side > 0 else l[t]) / p0 - 1)
            if worst <= 0.5:
                dn[i] = True; eq = max(worst, 0.0) if worst <= 0.05 else 0.5; break
            if best >= 1.5:
                up[i] = True; eq = 1.5; break
            eq = 1 + k * side * (c[t] / p0 - 1)
        end[i] = eq
    return up, dn, end


def row(name, up, dn, end):
    mid = ~up & ~dn
    p = up.mean()
    return (f"{name:<34} reached +50%: {p:5.1%} | lost 50% or more first: {dn.mean():5.1%} | neither: {mid.mean():5.1%} "
            f"(median {np.median(end[mid]) - 1 if mid.any() else float('nan'):+.0%}) | 3 months in a row: {p ** 3:6.2%} | 12 in a row: {p ** 12:.1e}")


if __name__ == "__main__":
    p = L.load_panel(fresh=True)
    blend = L.v4(p)[0]
    end_day = p["c"]["BTC"].dropna().index[-1]
    x = blend.loc[L.START:end_day].fillna(0.0).to_numpy()
    s = L.stats(pd.Series(x))
    m30 = pd.Series(x).add(1).rolling(DAYS).apply(np.prod, raw=True).dropna() - 1
    print(f"v4 on complete data at 1x, {len(x)} days: Sharpe {s['sharpe']:.2f}, yearly volatility {s['vol']:.0%}")
    print(f"30-day returns at 1x: median {m30.median():+.1%}, best 10% of months above {m30.quantile(.9):+.1%}, "
          f"worst 10% below {m30.quantile(.1):+.1%}, share of months at +50% or better: {float((m30 >= 0.5).mean()):.1%}\n")
    print("A. v4 with leverage, backtest edge taken at face value")
    for k in (1, 2, 3, 5, 8, 12):
        print("   " + row(f"{k}x", *play_returns(x, k)))
    print("B. v4 with leverage, if the edge is not real (same swings, zero average)")
    x0 = x - x.mean()
    for k in (3, 5, 8, 12):
        print("   " + row(f"{k}x", *play_returns(x0, k)))
    o, h, l, c = (vp.load(f, fresh=True)["BTC"].loc[L.START:end_day].to_numpy() for f in ("open", "high", "low", "close"))
    print("C. plain leveraged BTC, long or short at random (fees and funding not counted)")
    for k in (3, 5, 10, 20):
        ul, dl, el = play_btc(o, h, l, c, k, +1)
        us, ds, es = play_btc(o, h, l, c, k, -1)
        print("   " + row(f"{k}x", np.concatenate([ul, us]), np.concatenate([dl, ds]), np.concatenate([el, es])))
