"""v4 = 70% v3 trend book + 30% pump-follow sleeve (idiosyncratic short-term momentum, BTC-hedged).
See STRATEGY.md §8 for the honest caveats (contaminated OOS, survivorship, outlier dependence)."""
from __future__ import annotations
import numpy as np, pandas as pd
import engine, strategies as st, lab, target as tgt
from explore_v4 import load_all, build_v3, I
from robust_v4 import sleeve

W_V3, W_PUMP, THR, HOLD = 0.70, 0.30, 0.25, 5


def main():
    c, v, r = load_all(); e = st.eligibility(c, v, 60, 5e6, 30)
    v3 = engine.simulate(engine.vol_target(build_v3(c, v, e), r, I, target_vol=0.40), r, I)["net"]
    pump = sleeve(c, v, r, e, THR, HOLD)[0]["net"]
    net = W_V3 * v3 + W_PUMP * pump
    print(f"corr(v3, pump) = {v3.corr(pump):+.2f}\n")
    for name, x in [("v3 trend", v3), ("pump-follow", pump), ("v4 blend", net)]:
        for tag, seg in zip(("FULL", "IS", "OOS"), (x, *lab.split(x))):
            print(engine.fmt_metrics(engine.metrics(seg, I, f"{name} {tag}")))
        print()
    eq = (1 + net).cumprod()
    print("v4 yearly:", "  ".join(f"{y}:{g:+.0%}" for y, g in ((1 + net).groupby(net.index.year).prod() - 1).items()))
    ms = engine.monthly_summary(net)
    print(f"v4 monthly: mean {ms['mean_monthly']:.1%} median {ms['median_monthly']:.1%} "
          f"positive {ms['pct_positive']:.0%} worst {ms['worst']:.0%} best {ms['best']:.0%}")
    print("\nleverage (1yr block bootstrap w/ liquidation; IS-base = conservative)")
    a, b = lab.split(net)
    for k in (1.0, 1.5, 2.0, 3.0, 5.0):
        ri = tgt.block_bootstrap(a.dropna().to_numpy(), k, n_paths=3000)
        ro = tgt.block_bootstrap(b.dropna().to_numpy(), k, n_paths=3000)
        print(f"  {k:.1f}x  IS-base median {ri['median_monthly']:+.1%}/mo  P(lose half) {ri['p_lose_half']:.0%}  ruin {ri['p_ruin']:.0%}"
              f" | OOS-base {ro['median_monthly']:+.1%}/mo  P(lose half) {ro['p_lose_half']:.0%}")
    eq.rename("equity").to_csv("results/equity_v4.csv")


if __name__ == "__main__":
    main()
