"""
DEFINITIVE portfolio: 70% v3 (daily cross-sectional L/S trend, final_v3.py)
+ 30% CBS v2 (4h breakout swing, run_swing.py), static weights, rebalanced
implicitly by construction (sleeve returns weighted daily).

Why static 70/30 (all decided on IS <= 2024-12-31; see SWING_STRATEGY.md §8):
  * the static-mix dimension is a plateau (70/30..50/50: IS Sharpe 1.52-1.58
    on the common window) — no robust edge in the mix itself;
  * dynamic inverse-vol sleeve weights looked better IS (1.64) but overweight
    CBS exactly when it sits in cash (vol artifact) and lost OOS (1.31 vs
    1.43 static) — rejected;
  * blend-level vol targeting double-targets v3 (already vol-targeted
    internally) and only adds lag — rejected.

v3 here uses generic funding (real per-coin funding files are not in the
repo); its standalone numbers with real funding are in STRATEGY.md.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lab, engine
from final_v3 import build, EXCLUDE, I, TV
import swing_signals
from swing_engine import Config, run
from run_swing import windows, FINAL_SIG, FINAL_CFG

W_V3 = 0.70
SPLIT = pd.Timestamp("2024-12-31", tz="UTC")


def sleeve_returns():
    c, v, r, e = lab.load_data(I)
    e = e & ~e.columns.isin(EXCLUDE)
    v3 = engine.simulate(engine.vol_target(build(c, v, e), r, I, target_vol=TV),
                         r, I)["net"]
    base = swing_signals.Base()
    sig = base.build(**FINAL_SIG)
    i0, _, iT = windows(base)
    out = run(sig, Config(**FINAL_CFG), start=i0, end=iT)
    cbs = out["equity"].resample("1D").last().dropna().pct_change().dropna()
    common = v3.index.intersection(cbs.index)
    return v3[common], cbs[common]


def main():
    v3, cbs = sleeve_returns()
    b = (W_V3 * v3 + (1 - W_V3) * cbs).dropna()
    print(f"sleeve daily corr: {v3.corr(cbs):+.3f}   window: "
          f"{b.index[0].date()}..{b.index[-1].date()}\n")
    for tag, seg in [("IS  ", b[b.index <= SPLIT]), ("OOS ", b[b.index > SPLIT]),
                     ("FULL", b)]:
        print(engine.fmt_metrics(engine.metrics(seg, I, f"blend 70/30 {tag}")))
    print()
    for y in sorted(b.index.year.unique()):
        yr = b[b.index.year == y]
        if len(yr) > 60:
            print(engine.fmt_metrics(engine.metrics(yr, I, f"  {y}")))
    m = engine.monthly_returns(b)
    print(f"\nmonthly: mean={m.mean()*100:+.1f}% median={m.median()*100:+.1f}% "
          f"+{(m>0).mean()*100:.0f}% best={m.max()*100:+.1f}% worst={m.min()*100:+.1f}%")
    for k in (1.5, 2.0):
        lev = k * b - (k - 1) * 0.11 / 365     # extra gross pays ~funding
        mi = engine.metrics(lev[lev.index <= SPLIT], I, "")
        mo = engine.metrics(lev[lev.index > SPLIT], I, "")
        print(f"levered {k}x: IS Shrp={mi['Sharpe']:.2f} CAGR={mi['CAGR']*100:.0f}% "
              f"DD={mi['maxDD']*100:.0f}% | OOS Shrp={mo['Sharpe']:.2f} "
              f"CAGR={mo['CAGR']*100:.0f}% DD={mo['maxDD']*100:.0f}%")
    (1 + b.fillna(0)).cumprod().rename("equity").to_csv("results/blend_equity.csv")
    print("\nsaved results/blend_equity.csv")


if __name__ == "__main__":
    main()
