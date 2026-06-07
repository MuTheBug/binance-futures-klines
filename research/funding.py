"""Load Binance funding into a DAILY funding-rate matrix aligned to the kline grid,
and diagnose whether funding predicts returns (i.e., is there a carry edge & which sign)."""
from __future__ import annotations
import os, glob
import numpy as np, pandas as pd
import data, lab, engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FDIR = os.path.join(ROOT, "data", "funding")
CACHE = os.path.join(ROOT, "research", "cache", "funding_daily.parquet")


def build_daily_funding(daily_index):
    cols = {}
    for f in glob.glob(os.path.join(FDIR, "*_USDT_funding.csv")):
        base = os.path.basename(f)[: -len("_USDT_funding.csv")]
        df = pd.read_csv(f)
        if len(df) == 0:
            continue
        ts = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        day = ts.dt.floor("D")
        daily = df.assign(day=day).groupby("day")["fundingRate"].sum()   # sum of 3 events/day
        cols[base] = daily
    F = pd.DataFrame(cols).sort_index()
    F.to_parquet(CACHE)
    return F.reindex(daily_index)


def load_daily_funding(daily_index):
    if not os.path.exists(CACHE):
        return build_daily_funding(daily_index)
    F = pd.read_parquet(CACHE)
    return F.reindex(daily_index)


if __name__ == "__main__":
    c, v, r, e = lab.load_data("1d")
    F = build_daily_funding(c.index).reindex(columns=c.columns)
    cov = F.notna().sum()
    print(f"funding matrix: {F.shape}, symbols with data: {(cov>100).sum()}")
    ann = (F.mean() * 365)            # daily funding * 365 = annualized funding cost of a long
    print(f"annualized funding (median across coins): {ann.median()*100:.1f}%  "
          f"p10={ann.quantile(.1)*100:.1f}%  p90={ann.quantile(.9)*100:.1f}%")
    print("highest avg funding (most expensive longs):")
    print((ann.sort_values(ascending=False).head(6) * 100).round(1).to_string())

    # ---- predictive check: does trailing funding forecast forward returns? ----
    # cross-sectional Spearman corr between trailing-3d funding and NEXT-day return, by day
    Fbar = F.rolling(3, min_periods=2).mean().where(e)
    fwd = r.shift(-1).where(e)                         # next-day return (for diagnostic only)
    rows = []
    for t in c.index[200:]:
        a, b = Fbar.loc[t], fwd.loc[t]
        m = a.notna() & b.notna()
        if m.sum() >= 8:
            rows.append((a[m].rank().corr(b[m].rank())))
    ic = pd.Series(rows).dropna()
    print(f"\nfunding->fwd-return cross-sectional rank-IC: mean={ic.mean():.3f} "
          f"t-stat={ic.mean()/ic.std()*np.sqrt(len(ic)):.1f}  (negative => SHORT high-funding works)")
    # also the harvest view: avg funding sign
    print(f"share of (coin,day) with positive funding: {(F.where(e)>0).sum().sum()/F.where(e).notna().sum().sum()*100:.0f}%")
