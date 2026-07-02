"""IS-only robustness check of the aftershock effect: is it broad or driven
by a handful of monster bounces? Also scans gap/window definitions coarsely."""
import numpy as np
import pandas as pd

from seismo.engine import MIN_UNIVERSE, compute_signals
from seismo.panel import build_panel

IS_END = pd.Timestamp("2024-07-01", tz="UTC")
B_THR, NPICK, H = 0.85, 5, 24

panel = build_panel()
sig = compute_signals(panel)
idx = sig["z"].index
U, resid, z = sig["U"], sig["resid"], sig["z"]
RO = sig["RO"]


def basket_ret(t, hold=H):
    i = idx.get_loc(t)
    if i + 2 + hold >= len(idx):
        return np.nan
    univ = U.columns[U.loc[t].values]
    r = resid.loc[t, univ].dropna()
    if len(r) < NPICK:
        return np.nan
    picks = list(r.nsmallest(NPICK).index)
    legs = RO.iloc[i + 2: i + 2 + hold][picks].mean(axis=1)
    return (1 + legs).prod() - 1.0


def aftershocks(z_thr, gap_h, win_h):
    mask = (z <= -z_thr) & (sig["breadth_dn"] >= B_THR) & (sig["n_univ"] >= MIN_UNIVERSE)
    mask &= idx < IS_END
    raw = list(idx[mask])
    out = []
    for k, t in enumerate(raw):
        if k == 0:
            continue
        age = (t - raw[k - 1]).total_seconds() / 3600
        if gap_h < age <= win_h:
            out.append(t)
    return out


print("scan of (z_thr, gap, window) -> n / mean / median / %pos of 24h basket ret (IS):")
for z_thr in (2.0, 2.5, 3.0):
    for gap in (24, 48):
        for win in (72, 96, 120):
            evs = aftershocks(z_thr, gap, win)
            rets = pd.Series([basket_ret(t) for t in evs]).dropna()
            if len(rets) == 0:
                continue
            print(f"  z={z_thr} gap={gap:>3} win={win:>3}: n={len(rets):>3} "
                  f"mean={rets.mean() * 1e4:>6.0f}bps med={rets.median() * 1e4:>6.0f}bps "
                  f"pos={100 * (rets > 0).mean():>4.0f}%")

print("\nchosen definition z=2.5 gap=24 win=96 -- yearly breakdown + events:")
evs = aftershocks(2.5, 24, 96)
s = pd.Series({t: basket_ret(t) for t in evs}).dropna()
print(s.groupby(s.index.year).agg(["count", "mean", "median"]).round(4).to_string())
print("\nworst 5 / best 5 (unlevered basket, 24h):")
print(s.nsmallest(5).round(4).to_string())
print(s.nlargest(5).round(4).to_string())
