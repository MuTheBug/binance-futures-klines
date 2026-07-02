"""IS-only diagnostics: what separates cascades that bounce from ones that
keep falling? All conditioning analysis is restricted to data before
2024-07-01 (the out-of-sample fence)."""
import numpy as np
import pandas as pd

from seismo.engine import MIN_UNIVERSE, compute_signals
from seismo.panel import build_panel

IS_END = pd.Timestamp("2024-07-01", tz="UTC")
Z_THR, B_THR, NPICK = 2.5, 0.85, 5

panel = build_panel()
sig = compute_signals(panel)
idx = sig["z"].index
C, U, resid, m1, z = sig["C"], sig["U"], sig["resid"], sig["m1"], sig["z"]
RO = sig["RO"]

mask = (z <= -Z_THR) & (sig["breadth_dn"] >= B_THR) & (sig["n_univ"] >= MIN_UNIVERSE)
mask &= idx < IS_END
raw = list(idx[mask])

# de-duplicate: one event per 48h cluster, keep the FIRST trigger
events, last = [], None
for t in raw:
    if last is None or (t - last) > pd.Timedelta(hours=48):
        events.append(t)
    last = t
print(f"IS clustered events: {len(events)}")

# market index for conditioning
mkt = (1 + m1.fillna(0)).cumprod()
mkt_ma = mkt.rolling(720, min_periods=500).mean()
mkt_dd7 = mkt / mkt.rolling(24 * 7, min_periods=24).max() - 1.0

rows, paths = [], []
H = 48
for t in events:
    i = idx.get_loc(t)
    if i + 2 + H >= len(idx):
        continue
    univ = U.columns[U.loc[t].values]
    r = resid.loc[t, univ].dropna()
    if len(r) < NPICK:
        continue
    picks = list(r.nsmallest(NPICK).index)
    legs = RO.iloc[i + 2: i + 2 + H][picks].mean(axis=1)
    cum = (1 + legs).cumprod().values - 1.0
    prev = [e for e in raw if e < t]
    hrs_since_prev = (t - prev[-1]).total_seconds() / 3600 if prev else np.inf
    rows.append({
        "event": t,
        "z": z.loc[t],
        "trend_up": mkt.loc[t] > mkt_ma.loc[t],
        "dd7": mkt_dd7.loc[t],
        "hrs_since_prev": hrs_since_prev,
        "ret6": cum[5], "ret12": cum[11], "ret24": cum[23], "ret48": cum[47],
    })
    paths.append(cum)

df = pd.DataFrame(rows)
paths = np.array(paths)

print("\nmean cumulative basket return (bps) by hour-in-trade:")
prof = paths.mean(axis=0) * 1e4
for h in [1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48]:
    print(f"  h={h:>2}: {prof[h - 1]:>7.1f}")

def bucket(name, series, splits):
    print(f"\n--- by {name} ---")
    lab = pd.cut(series, splits)
    g = df.groupby(lab, observed=True)[["ret12", "ret24", "ret48"]]
    print((g.mean() * 1e4).round(0).join(g.size().rename("n")).to_string())

bucket("z severity", df["z"], [-15, -5, -4, -3, -2.5])
bucket("mkt drawdown from 7d high", df["dd7"], [-1, -0.15, -0.08, -0.04, 0])
bucket("hours since previous trigger", df["hrs_since_prev"],
       [0, 72, 24 * 14, np.inf])

print("\n--- by 30d trend ---")
g = df.groupby("trend_up")[["ret12", "ret24", "ret48"]]
print((g.mean() * 1e4).round(0).join(g.size().rename("n")).to_string())
