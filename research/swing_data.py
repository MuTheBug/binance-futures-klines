"""
Data layer for the 4h swing strategy.

Builds 4h OHLCV matrices by resampling the cached 1h matrices (UTC-aligned
00/04/08/12/16/20 boundaries) and caches them as parquet. Daily matrices come
straight from data.py's cache (built from the 1d CSVs, history back to 2020)
and are used for the regime filter and point-in-time universe.

Conventions:
  * index = bar OPEN time (UTC). A 4h bar with open time T covers (T, T+4h];
    its close is known at T+4h.
  * open  = first non-NaN 1h open in the window, high = max, low = min,
    close = last, volume = sum.
"""
from __future__ import annotations
import os
import pandas as pd
import data

CACHE = data.CACHE
FIELDS = data.FIELDS

# tokenized stocks / metals / oil / FX proxies — not crypto; they list late
# (2025-26 only) and inflate OOS results, so they are excluded everywhere.
NON_CRYPTO = {
    "NVDA", "TSLA", "MSTR", "SPY", "QQQ", "INTC", "MU", "MRVL", "AMD", "SNDK",
    "SKHYNIX", "NOK", "EWY", "SOXL", "CRCL", "SPCX", "XAG", "XAU", "XAUT",
    "PAXG", "CL", "BZ", "SLX", "BILL", "COIN", "GENIUS", "CHIP", "BTW", "ZEST",
}

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def ensure_1h_cache():
    if not os.path.exists(os.path.join(CACHE, "close_1h.parquet")):
        print("building 1h matrices (one-time) ...")
        data.build_matrices("1h", verbose=True)
    if not os.path.exists(os.path.join(CACHE, "close_1d.parquet")):
        print("building 1d matrices (one-time) ...")
        data.build_matrices("1d", verbose=True)


def build_4h(force: bool = False):
    """Resample cached 1h matrices to 4h and cache."""
    ensure_1h_cache()
    if not force and os.path.exists(os.path.join(CACHE, "close_4h.parquet")):
        return
    print("resampling 1h -> 4h ...")
    for fld in FIELDS:
        m = data.load(fld, "1h")
        r = m.resample("4h", label="left", closed="left").agg(_AGG[fld])
        if fld == "volume":               # windows with no data must stay NaN, not 0
            has = data.load("close", "1h").resample("4h", label="left", closed="left").last()
            r = r.where(has.notna())
        out = os.path.join(CACHE, f"{fld}_4h.parquet")
        rr = r.copy()
        # store as ms epoch like the other caches (robust to pandas' index resolution)
        ms = rr.index.tz_convert("UTC").tz_localize(None).astype("datetime64[ms]").astype("int64")
        rr.index = pd.Index(ms, name="timestamp")
        rr.to_parquet(out)
        print(f"  wrote {out} shape={r.shape}")


def load_4h() -> dict[str, pd.DataFrame]:
    build_4h()
    out = {fld: data.load(fld, "4h") for fld in FIELDS}
    # drop columns that are entirely NaN and non-crypto symbols
    keep = [c for c in out["close"].columns
            if c not in NON_CRYPTO and out["close"][c].notna().any()]
    return {f: m[keep] for f, m in out.items()}


def load_daily() -> dict[str, pd.DataFrame]:
    ensure_1h_cache()
    out = {fld: data.load(fld, "1d") for fld in FIELDS}
    keep = [c for c in out["close"].columns if c not in NON_CRYPTO]
    return {f: m[keep] for f, m in out.items()}


if __name__ == "__main__":
    build_4h(force=True)
    m4 = load_4h()
    print("4h close:", m4["close"].shape, m4["close"].index[0], "->", m4["close"].index[-1])
    md = load_daily()
    print("1d close:", md["close"].shape, md["close"].index[0], "->", md["close"].index[-1])
