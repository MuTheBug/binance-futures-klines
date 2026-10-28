"""Robustness gauntlet for the pump-FOLLOW sleeve (H2', idiosyncratic short-term momentum).
CAVEAT: this hypothesis is the sign-flip of H2 (pump-fade), whose OOS we already saw -> the
OOS here is CONTAMINATED. Treat every number below as in-sample-grade evidence."""
from __future__ import annotations
import numpy as np, pandas as pd
import engine, strategies as st, lab
from explore_v4 import load_all, h2_pump, build_v3, I


def sleeve(c, v, r, e, thr, hold, delay=0, cost=engine.COST_PER_SIDE):
    w, F = h2_pump(c, v, r, thr, hold, e)
    w, F = -w.shift(delay).fillna(0.0), -F                       # follow; longs pay stress funding
    wt = engine.vol_target(w, r, I, target_vol=0.40, cost_per_side=cost, funding_matrix=F)
    return engine.simulate(wt, r, I, cost_per_side=cost, funding_matrix=F), wt


def sh(x):
    return engine.metrics(x, I)["Sharpe"]


def main():
    c, v, r = load_all(); e = st.eligibility(c, v, 60, 5e6, 30)
    v3 = engine.simulate(engine.vol_target(build_v3(c, v, e), r, I, target_vol=0.40), r, I)["net"]

    print("1) parameter grid, Sharpe IS / OOS (sleeve alone)")
    holds = (2, 3, 5, 7, 10, 15)
    print("   thr\\hold " + " ".join(f"{h:>11}" for h in holds))
    for thr in (0.15, 0.20, 0.25, 0.30, 0.35):
        row = []
        for h in holds:
            n = sleeve(c, v, r, e, thr, h)[0]["net"]; a, b = lab.split(n)
            row.append(f"{sh(a):>5.2f}/{sh(b):>5.2f}")
        print(f"   {thr:>8.2f} " + " ".join(row))

    base, wt = sleeve(c, v, r, e, 0.25, 5)
    n = base["net"]
    print("\n2) cost stress (thr .25 hold 5), Sharpe IS/OOS")
    for cps in (0.0007, 0.0015, 0.0030):
        x = sleeve(c, v, r, e, 0.25, 5, cost=cps)[0]["net"]; a, b = lab.split(x)
        print(f"   {cps*1e4:>4.0f} bps/side: {sh(a):.2f} / {sh(b):.2f}")
    print("\n3) entry delay (miss the close, enter 1-2 days late)")
    for d in (1, 2):
        x = sleeve(c, v, r, e, 0.25, 5, delay=d)[0]["net"]; a, b = lab.split(x)
        print(f"   delay {d}d: {sh(a):.2f} / {sh(b):.2f}")
    print("\n4) dependence on outliers: drop the N best days")
    for N in (5, 10, 20):
        x = n.copy(); x[x.nlargest(N).index] = 0.0; a, b = lab.split(x)
        print(f"   drop top {N:>2}: {sh(a):.2f} / {sh(b):.2f}")
    print("\n5) tail risk")
    print(f"   worst day {n.min()*100:.1f}%  best day {n.max()*100:.1f}%  "
          f"days in market {(wt.abs().sum(1)>0).mean()*100:.0f}%  max gross {wt.abs().sum(1).max():.1f}x")
    print("   worst 5 days:", ", ".join(f"{t.date()} {x*100:+.1f}%" for t, x in n.nsmallest(5).items()))
    print("\n6) yearly sleeve returns:", "  ".join(f"{y}:{g:+.0%}" for y, g in
          ((1 + n).groupby(n.index.year).prod() - 1).items()))


if __name__ == "__main__":
    main()
