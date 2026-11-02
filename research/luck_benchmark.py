"""What does "loop until the backtest is spectacular" produce? A luck benchmark on this repo's data.

Family A, NOISE: 1000 strategies whose signals are random numbers (smoothed so their turnover looks
           realistic). Their true edge is ZERO by construction.
Family B, INDICATORS: 1000 random but plausible-looking rules (momentum / reversal / volatility /
           volume / distance-from-high, random lookback, sign, cross-sectional or time-series,
           random smoothing).
Same machinery as the real strategies: crypto-only universe, inverse-vol sizing, 40% vol target,
14 bps round-trip costs, funding drag. IS = 2020-06..2024-12, OOS = 2025-01..2026-06.
Question: if you keep the best in-sample result, what happens out of sample?
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from statistics import NormalDist
import engine
import lab
from final_v3 import EXCLUDE

I = "1d"
N = 1000
rng = np.random.default_rng(20261102)


def setup():
    c, v, r, e = lab.load_data(I)
    e = e & ~e.columns.isin(EXCLUDE)
    iv = (1.0 / r.rolling(15, min_periods=8).std()).where(e)
    return c, v, r, e, iv


def run(signal, r, e, iv):
    w = (signal * iv).where(e, 0.0).fillna(0.0)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    net = engine.simulate(engine.vol_target(w, r, I, target_vol=0.40), r, I)["net"]
    a, b = lab.split(net)
    m = lambda x: engine.metrics(x, I)
    return m(a), m(b), m(net)


def noise_signal(shape, idx, cols):
    span = int(rng.integers(1, 31))
    z = pd.DataFrame(rng.standard_normal(shape), index=idx, columns=cols)
    return z.ewm(span=span).mean(), f"noise ema{span}"


def indicator_signal(c, v, r, feats):
    name = rng.choice(list(feats))
    L = int(rng.choice([2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120]))
    raw = feats[name](L)
    sign = int(rng.choice([-1, 1]))
    mode = rng.choice(["xs", "ts"])
    if mode == "xs":
        sig = raw.rank(axis=1, pct=True) - 0.5
    else:
        sig = np.tanh(raw.div(raw.rolling(90, min_periods=30).std()))
    span = int(rng.integers(1, 21))
    sig = (sign * sig).ewm(span=span).mean()
    if rng.random() < 0.3:
        sig = sig.clip(lower=0)                         # long-only variant
    return sig, f"{name}{L} {'+' if sign > 0 else '-'} {mode} ema{span}"


def main(summary_only=False):
    c, v, r, e, iv = setup()
    feats = {
        "ret": lambda L: c.pct_change(L, fill_method=None),
        "vol": lambda L: r.rolling(L, min_periods=max(2, L // 2)).std(),
        "volu": lambda L: np.log(v.rolling(L, min_periods=max(2, L // 2)).mean() / v.rolling(120, min_periods=30).mean()),
        "dhigh": lambda L: c / c.rolling(L, min_periods=max(2, L // 2)).max() - 1,
        "skew": lambda L: r.rolling(max(L, 5), min_periods=5).skew(),
    }
    rows = []
    for fam in (() if summary_only else ("NOISE", "INDICATORS")):
        for i in range(N):
            sig, desc = noise_signal(c.shape, c.index, c.columns) if fam == "NOISE" else indicator_signal(c, v, r, feats)
            mi, mo, mf = run(sig, r, e, iv)
            rows.append((fam, desc, mi["Sharpe"], mi["CAGR"], mo["Sharpe"], mo["CAGR"], mf["Sharpe"], mf["CAGR"], mf["maxDD"]))
            if (i + 1) % 250 == 0:
                print(f"  {fam}: {i + 1}/{N}", flush=True)
    if summary_only:
        df = pd.read_csv("results/luck_benchmark.csv")
    else:
        df = pd.DataFrame(rows, columns=["family", "desc", "is_sh", "is_cagr", "oos_sh", "oos_cagr", "all_sh", "all_cagr", "all_dd"])
        df.to_csv("results/luck_benchmark.csv", index=False)

    yrs_is = len(lab.split(r.iloc[:, 0])[0]) / 365
    for fam, g in df.groupby("family", sort=False):
        print(f"\n===== {fam}: {len(g)} strategies =====")
        b = g.loc[g.is_sh.idxmax()]
        print(f"  best IN-SAMPLE      : {b.desc:<28} IS Sharpe {b.is_sh:5.2f}  CAGR {b.is_cagr:7.1%}  "
              f"-> OUT-OF-SAMPLE Sharpe {b.oos_sh:5.2f}  CAGR {b.oos_cagr:7.1%}")
        top = g.nlargest(10, "is_sh")
        print(f"  top-10 in-sample    : mean IS Sharpe {top.is_sh.mean():5.2f}  -> mean OOS Sharpe {top.oos_sh.mean():5.2f}"
              f"  (OOS CAGR {top.oos_cagr.mean():7.1%})")
        f_ = g.loc[g.all_sh.idxmax()]
        print(f"  best on ALL data    : {f_.desc:<28} Sharpe {f_.all_sh:5.2f}  CAGR {f_.all_cagr:7.1%}  maxDD {f_.all_dd:6.1%}"
              f"   <- 'loop until spectacular'; no data left to test it")
        rho = g.is_sh.rank().corr(g.oos_sh.rank())          # Spearman without scipy
        print(f"  rank correlation IS vs OOS Sharpe: {rho:+.2f}")
    # expected best Sharpe from pure luck (Bailey & Lopez de Prado, False Strategy Theorem)
    sd = np.sqrt(1 / yrs_is)
    z = NormalDist().inv_cdf
    em = lambda n: sd * ((1 - 0.5772) * z(1 - 1 / n) + 0.5772 * z(1 - 1 / (n * np.e)))
    print(f"\nexpected best IS Sharpe from luck alone ({yrs_is:.1f} yrs of data): "
          + "  ".join(f"N={n}: {em(n):.2f}" for n in (10, 100, 350, 1000, 10000)))


if __name__ == "__main__":
    import sys
    main(summary_only="summary" in sys.argv)
