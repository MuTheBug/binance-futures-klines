"""Part A of V5_PROTOCOL.md: how biased were the published v3/v4 numbers?

Same rules, four data setups:
  1. old 156-symbol universe, assumed funding  (should reproduce final_v4.py)
  2. old universe, real funding
  3. full universe (incl. delisted), assumed funding
  4. full universe, real funding               <- what the bot would actually have faced
Design window only (<= 2026-06-07). IS = 2020-06-01..2024-12-31, VAL = 2025-01-01..2026-06-07.
"""
import numpy as np
import pandas as pd

import lab
import strategies as st
import v5_lib as L
import vision_panel as vp
from final_v3 import EXCLUDE


def old_panel():
    c, v, r, e = lab.load_data("1d")
    keep = [x for x in c.columns if x not in EXCLUDE]
    c, v = c[keep].astype("float64"), v[keep].astype("float64")
    e = st.eligibility(c, v, 60, 5e6, 30)
    F = vp.load("funding").reindex(index=c.index, columns=c.columns)
    return dict(c=c, r=c.pct_change(fill_method=None), e=e, F=F)


def show(tag, p, F):
    blend, n3, npump = L.v4(p, F=F)
    print(f"--- {tag}")
    for name, x in (("v3 trend", n3), ("pump-follow", npump), ("v4 blend 70/30", blend)):
        print("   " + L.fmt(name, x))
    return blend, n3, npump


if __name__ == "__main__":
    po = old_panel()
    pf = L.load_panel()
    print(f"old universe: {po['c'].shape[1]} coins, median eligible {int(po['e'].sum(1).loc[L.START:].median())} | "
          f"full universe: {pf['c'].shape[1]} instruments, median eligible {int(pf['e'].sum(1).loc[L.START:].median())}")
    fcov = pf["F"].notna() | ~pf["e"]
    print(f"real funding available on {float((pf['F'].notna() & pf['e']).sum().sum() / pf['e'].sum().sum()):.1%} of eligible coin-days")
    res = {}
    res[1] = show("1. old universe, assumed funding (published setup)", po, "generic")
    res[2] = show("2. old universe, real funding", po, "real")
    res[3] = show("3. full universe incl. delisted, assumed funding", pf, "generic")
    res[4] = show("4. full universe incl. delisted, real funding", pf, "real")
    b = res[4][0]
    yr = {y: round(L.stats(g)["sharpe"], 2) for y, g in b.loc[L.START:].groupby(b.loc[L.START:].index.year)}
    print("\nsetup 4, v4 blend Sharpe by year:", yr)
    print("setup 4 vs setup 1 daily-return correlation (v4 blend):", round(float(res[4][0].corr(res[1][0])), 2))
    # what the real funding looked like for the positions actually held
    c, r, e, F = pf["c"], pf["r"], pf["e"], pf["F"]
    wp, held = L.pump_weights(c, r, e)
    f_held = F.where(held).stack() if hasattr(F.where(held), "stack") else None
    fh = F.where(held).to_numpy().ravel(); fh = fh[np.isfinite(fh)]
    print(f"pump sleeve longs: real funding paid, annualised: median {np.median(fh) * 365:+.0%}, mean {np.mean(fh) * 365:+.0%}, "
          f"p10 {np.quantile(fh, .1) * 365:+.0%}, p90 {np.quantile(fh, .9) * 365:+.0%}  (old assumption: +50%)")
    n_events_old = int(((po['r'].sub(po['r']['BTC'], axis=0) > 0.25) & po['e']).loc[L.START:].sum().sum())
    n_events_new = int(((r.sub(r['BTC'], axis=0) > 0.25) & e).loc[L.START:].sum().sum())
    print(f"pump triggers since 2020-06: old universe {n_events_old}, full universe {n_events_new}")
    pd.DataFrame({f"setup{k}_{n}": v[i] for k, v in res.items() for i, n in enumerate(("blend", "v3", "pump"))}).to_pickle("results/v5_audit.pkl")
