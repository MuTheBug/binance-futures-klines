"""
Data layer: audit raw CSVs and build cached, aligned OHLCV matrices.

Raw files live in the repo root as <BASE>_USDT_<interval>.csv with columns:
    timestamp(ms), open, high, low, close, volume, datetime

We build, for each interval, aligned matrices (index=timestamp, columns=symbol)
for open/high/low/close/volume, cached as parquet under research/cache/.
Symbols that list late simply have NaN before their first bar.
"""
from __future__ import annotations
import os, glob, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
CACHE = os.path.join(ROOT, "research", "cache")
os.makedirs(CACHE, exist_ok=True)

FIELDS = ["open", "high", "low", "close", "volume"]


def _files(interval: str):
    pat = os.path.join(ROOT, f"*_USDT_{interval}.csv")
    out = {}
    for f in glob.glob(pat):
        base = os.path.basename(f)[: -len(f"_USDT_{interval}.csv")]
        out[base] = f
    return out


def audit(interval: str) -> pd.DataFrame:
    """Per-symbol data quality report."""
    rows = []
    step_ms = 86_400_000 if interval == "1d" else 3_600_000
    for base, f in sorted(_files(interval).items()):
        try:
            df = pd.read_csv(f, usecols=["timestamp", "open", "high", "low", "close", "volume"])
        except Exception as e:
            rows.append({"symbol": base, "error": str(e)})
            continue
        if len(df) == 0:
            rows.append({"symbol": base, "n": 0})
            continue
        ts = df["timestamp"].to_numpy()
        c = df["close"].to_numpy(dtype=float)
        v = df["volume"].to_numpy(dtype=float)
        dupes = len(ts) - len(np.unique(ts))
        span = (ts.max() - ts.min()) / step_ms + 1
        gap_ratio = 1.0 - len(df) / span if span > 0 else np.nan
        dollar_vol = np.nanmedian(c * v)
        rows.append({
            "symbol": base,
            "n": len(df),
            "start": pd.to_datetime(ts.min(), unit="ms").strftime("%Y-%m-%d"),
            "end": pd.to_datetime(ts.max(), unit="ms").strftime("%Y-%m-%d"),
            "dupes": int(dupes),
            "gap_ratio": round(float(gap_ratio), 4),
            "n_nan_close": int(np.isnan(c).sum()),
            "n_nonpos_close": int((c <= 0).sum()),
            "med_dollar_vol": float(dollar_vol),
        })
    return pd.DataFrame(rows)


def build_matrices(interval: str, verbose=True):
    """Build aligned OHLCV matrices on a regular time grid; cache as parquet."""
    files = _files(interval)
    step_ms = 86_400_000 if interval == "1d" else 3_600_000
    # collect per-symbol frames
    frames = {fld: {} for fld in FIELDS}
    gmin, gmax = None, None
    for i, (base, f) in enumerate(sorted(files.items())):
        df = pd.read_csv(f, usecols=["timestamp"] + FIELDS)
        df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        if len(df) == 0:
            continue
        for fld in FIELDS:
            frames[fld][base] = df[fld].astype("float32")
        gmin = df.index.min() if gmin is None else min(gmin, df.index.min())
        gmax = df.index.max() if gmax is None else max(gmax, df.index.max())
        if verbose and i % 50 == 0:
            print(f"  read {i}/{len(files)} {interval}", file=sys.stderr)
    grid = np.arange(gmin, gmax + step_ms, step_ms)
    for fld in FIELDS:
        mat = pd.DataFrame(frames[fld]).reindex(grid)
        mat.index.name = "timestamp"
        out = os.path.join(CACHE, f"{fld}_{interval}.parquet")
        mat.to_parquet(out)
        if verbose:
            print(f"  wrote {out}  shape={mat.shape}", file=sys.stderr)
    return grid


def load(field: str, interval: str) -> pd.DataFrame:
    """Load a cached matrix with a DatetimeIndex (UTC)."""
    p = os.path.join(CACHE, f"{field}_{interval}.parquet")
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index.astype("int64"), unit="ms", utc=True)
    return df


if __name__ == "__main__":
    for interval in ["1d", "1h"]:
        print(f"\n===== AUDIT {interval} =====")
        a = audit(interval)
        a.to_csv(os.path.join(ROOT, "research", "results", f"audit_{interval}.csv"), index=False)
        # summary
        full = a[a["n"] > 0]
        print(f"symbols: {len(a)}  with_data: {len(full)}")
        print(f"history rows: min={full['n'].min()} median={int(full['n'].median())} max={full['n'].max()}")
        print(f"date span: {full['start'].min()} .. {full['end'].max()}")
        print(f"dupes>0: {(full['dupes']>0).sum()}  nan_close>0: {(full['n_nan_close']>0).sum()}  nonpos_close>0: {(full['n_nonpos_close']>0).sum()}")
        print(f"gap_ratio: median={full['gap_ratio'].median():.4f} p90={full['gap_ratio'].quantile(.9):.4f} max={full['gap_ratio'].max():.4f}")
        print("top 15 by median dollar volume:")
        print(full.sort_values("med_dollar_vol", ascending=False).head(15)[["symbol","n","start","med_dollar_vol"]].to_string(index=False))
        print(f"\nbuilding matrices for {interval} ...")
        build_matrices(interval)
