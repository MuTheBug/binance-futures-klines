"""Post-exam diagnostics (run AFTER results/v5_fresh_result.json exists). Explanation only: nothing
here may be used to re-tune anything, because FRESH is no longer unseen.

1. How unusual is the FRESH result against the backtest's own 5-month windows?
2. Where did the loss come from (trend vs pump, longs vs shorts, which days and coins)?
"""
import os
import sys

import numpy as np
import pandas as pd

import v5_lib as L
import vision_panel as vp
from v5_exam import FRESH_START, RESULT

if not os.path.exists(RESULT):
    sys.exit("run the exam first")
p = L.load_panel(fresh=True)
c, r, e, F = p["c"], p["r"], p["e"], p["F"]
blend, n3, npump = L.v4(p)
end = c["BTC"].dropna().index[-1]
fresh = slice(pd.Timestamp(FRESH_START, tz="UTC"), end)
n = len(blend.loc[fresh])
print(f"FRESH window: {n} days")
for name, x in (("v4 blend (idealized, 1x)", blend), ("trend part", n3), ("pump part", npump)):
    design = x.loc[L.START:L.DESIGN_END]
    roll = (1 + design).rolling(n).apply(np.prod, raw=True).dropna() - 1
    got = float((1 + x.loc[fresh]).prod() - 1)
    print(f"{name:<26} FRESH {got:+6.1%} | design-window {n}-day windows: median {roll.median():+6.1%}, 10th pct {roll.quantile(.1):+6.1%}, "
          f"worst {roll.min():+6.1%} | share of windows this bad or worse: {float((roll <= got).mean()):.1%}")
    worst = roll.idxmin()
    print(f"{'':<26} worst design window ended {worst.date()}; windows below 0: {float((roll < 0).mean()):.0%}")

# where the FRESH loss came from
w3 = L.v3_weights(c, e); _, W3 = L.run(w3, r, L.COST, F)
wp, held = L.pump_weights(c, r, e); _, WP = L.run(wp, r, L.COST, F)
R = r.fillna(0)
for name, W in (("trend", W3), ("pump", WP)):
    Wp = W.shift(1).fillna(0).loc[fresh]
    pl, ps = (Wp.clip(lower=0) * R.loc[fresh]).sum().sum(), (Wp.clip(upper=0) * R.loc[fresh]).sum().sum()
    print(f"\n{name}: P&L from longs {pl:+.3f}, from shorts {ps:+.3f}; avg net exposure {Wp.sum(1).mean():+.2f}, avg gross {Wp.abs().sum(1).mean():.2f}")
    contrib = (Wp * R.loc[fresh]).sum().sort_values()
    print("   worst coins:", {k: round(v, 3) for k, v in contrib.head(6).items()}, "| best:", {k: round(v, 3) for k, v in contrib.tail(4).items()})
    m = (Wp * R.loc[fresh]).sum(1)
    print("   net exposure by month:", {f"{t:%m}": round(v, 2) for t, v in Wp.sum(1).resample("ME").mean().items()})
    print("   5 worst days:", [(str(t.date()), round(v, 3)) for t, v in m.nsmallest(5).items()])
bm = pd.DataFrame({"BTC": r["BTC"], "eq-wt coins": r.where(e.shift(1, fill_value=False)).mean(1), "v4 1x": blend,
                   "trend": n3, "pump": npump}).loc[fresh]
print("\nmonthly returns:")
print(((1 + bm).resample("ME").prod() - 1).round(3).to_string())
aug = bm.loc[pd.Timestamp("2026-08-01", tz="UTC"):pd.Timestamp("2026-08-31", tz="UTC")]
print("\nAugust 2026, worst 6 days for v4:"); print(aug.loc[aug["v4 1x"].nsmallest(6).index].sort_index().round(3).to_string())
