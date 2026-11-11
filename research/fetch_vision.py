"""Download Binance USD-M futures history from data.binance.vision (public bulk data).

Why: the repo's CSVs hold 156 symbols that were alive (and selected) in June 2026. That is a
survivorship- and selection-biased universe with no funding or positioning data. This script builds
the complete one: every USDT perpetual ever listed (incl. delisted), real funding, and positioning
metrics (open interest, long/short ratios).

    python3 fetch_vision.py symbols            # list symbols + available months (metadata only)
    python3 fetch_vision.py klines 1d          # daily klines, all symbols, all months + current month
    python3 fetch_vision.py funding            # funding-rate events
    python3 fetch_vision.py metrics            # daily aggregates of 5-min positioning metrics
                                               # (only symbol-days listed in data/vision/metrics_plan.csv)
Output: data/vision/ (git-ignored). Needs network access to data.binance.vision and the bucket
listing at s3-ap-northeast-1.amazonaws.com.
"""
from __future__ import annotations
import io
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "vision")
BASE = "https://data.binance.vision/"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
UM = "data/futures/um/"
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
         "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
_local = None


def session() -> requests.Session:
    global _local
    import threading
    if _local is None:
        _local = threading.local()
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def get(url, tries=5):
    for a in range(tries):
        try:
            r = session().get(url, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r.content
            err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            err = repr(e)
        time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"{url}: {err}")


def list_prefix(prefix: str):
    """All (common prefixes, keys) directly under an S3 prefix (delimiter '/'), paginated."""
    prefixes, keys, marker = [], [], ""
    while True:
        url = f"{S3}?delimiter=/&prefix={prefix}" + (f"&marker={marker}" if marker else "")
        x = get(url).decode()
        prefixes += re.findall(r"<CommonPrefixes><Prefix>([^<]+)</Prefix>", x)
        ks = re.findall(r"<Key>([^<]+)</Key>", x)
        keys += ks
        if "<IsTruncated>true" not in x:
            return prefixes, keys
        nm = re.search(r"<NextMarker>([^<]+)</NextMarker>", x)
        marker = nm.group(1) if nm else (ks[-1] if ks else prefixes[-1])


def read_zip_csv(content: bytes, names=None) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(content))
    raw = z.read(z.namelist()[0])
    head = raw[:64].decode(errors="ignore")
    has_header = not head[:1].isdigit()
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, names=None if has_header else names)


def pmap(fn, items, workers=32, label=""):
    out, t0 = [], time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fn, it) for it in items]
        for i, f in enumerate(as_completed(futs)):
            out.append(f.result())
            if (i + 1) % 2000 == 0:
                print(f"  {label} {i + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    return out


# ------------------------------------------------------------------------------------ symbols
def cmd_symbols():
    os.makedirs(OUT, exist_ok=True)
    pm, _ = list_prefix(UM + "monthly/klines/")
    pd_, _ = list_prefix(UM + "daily/klines/")
    syms = sorted({p.rstrip("/").split("/")[-1] for p in pm + pd_})
    print(f"{len(syms)} symbols in the bucket (monthly {len(pm)}, daily {len(pd_)})")

    def months(sym):
        _, keys = list_prefix(f"{UM}monthly/klines/{sym}/1d/")
        ms = sorted(re.findall(r"-1d-(\d{4}-\d{2})\.zip$", k)[0] for k in keys if k.endswith(".zip"))
        return sym, ms
    rows = pmap(months, syms, workers=24, label="list")
    df = pd.DataFrame([{"symbol": s, "first_month": m[0] if m else "", "last_month": m[-1] if m else "",
                        "n_months": len(m), "months": " ".join(m)} for s, m in rows]).sort_values("symbol")
    df.to_csv(os.path.join(OUT, "symbols.csv"), index=False)
    print(df[["symbol", "first_month", "last_month", "n_months"]].describe(include="all").to_string())


def perp_symbols() -> pd.DataFrame:
    """USDT-margined perpetuals, incl. the *SETTLED histories Binance keeps for contracts that were
    delisted and later relisted under the same name (kept as separate instruments)."""
    df = pd.read_csv(os.path.join(OUT, "symbols.csv"), dtype=str).fillna("")
    ok = df.symbol.str.match(r"^[^_]+USDT(_?SETTLED)*$")
    return df[ok].reset_index(drop=True)


# ------------------------------------------------------------------------------------ klines
def cmd_klines(interval="1d", last_days=45):
    df = perp_symbols()
    today = pd.Timestamp.now(tz="UTC").floor("D")
    recent = [str((today - pd.Timedelta(days=i)).date()) for i in range(1, last_days + 1)]
    out_dir = os.path.join(OUT, f"klines_{interval}")
    os.makedirs(out_dir, exist_ok=True)

    def one(row):
        sym, months = row.symbol, row.months.split()
        parts = []
        for m in months:
            c = get(f"{BASE}{UM}monthly/klines/{sym}/{interval}/{sym}-{interval}-{m}.zip")
            if c:
                parts.append(read_zip_csv(c, KCOLS))
        last_m = months[-1] if months else "0000-00"
        for d in recent:                      # days after the last monthly file
            if d[:7] > last_m:
                c = get(f"{BASE}{UM}daily/klines/{sym}/{interval}/{sym}-{interval}-{d}.zip")
                if c:
                    parts.append(read_zip_csv(c, KCOLS))
        if not parts:
            return sym, 0
        k = pd.concat(parts, ignore_index=True)
        k["open_time"] = pd.to_numeric(k["open_time"])
        k.loc[k.open_time > 1e14, "open_time"] //= 1000             # microseconds -> ms (defensive)
        k = k.drop_duplicates("open_time").sort_values("open_time")
        k = k[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "count",
               "taker_buy_volume", "taker_buy_quote_volume"]].astype("float64")
        k["open_time"] = k["open_time"].astype("int64")
        k.to_parquet(os.path.join(out_dir, f"{sym}.parquet"), index=False)
        return sym, len(k)
    res = pmap(one, list(df.itertuples()), workers=16, label=f"klines {interval}")
    n = pd.Series(dict(res))
    print(f"klines {interval}: {int((n > 0).sum())} symbols, {int(n.sum())} bars")


# ------------------------------------------------------------------------------------ funding
def cmd_funding():
    df = perp_symbols()
    out_dir = os.path.join(OUT, "funding")
    os.makedirs(out_dir, exist_ok=True)

    def one(row):
        sym = row.symbol
        _, keys = list_prefix(f"{UM}monthly/fundingRate/{sym}/")
        parts = []
        for k in sorted(k for k in keys if k.endswith(".zip")):
            c = get(BASE + k)
            if c:
                parts.append(read_zip_csv(c, ["calc_time", "funding_interval_hours", "last_funding_rate"]))
        if not parts:
            return sym, 0
        f = pd.concat(parts, ignore_index=True).drop_duplicates("calc_time").sort_values("calc_time")
        f.to_parquet(os.path.join(out_dir, f"{sym}.parquet"), index=False)
        return sym, len(f)
    res = pmap(one, list(df.itertuples()), workers=16, label="funding")
    n = pd.Series(dict(res))
    print(f"funding: {int((n > 0).sum())} symbols, {int(n.sum())} events")


# ------------------------------------------------------------------------------------ metrics
MCOLS = ["sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
         "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]


def metrics_day(sym, day):
    c = get(f"{BASE}{UM}daily/metrics/{sym}/{sym}-metrics-{day}.zip")
    if c is None:
        return None
    try:
        z = zipfile.ZipFile(io.BytesIO(c))
        lines = z.read(z.namelist()[0]).decode().splitlines()[1:]
        try:                                                # fast path: plain numeric rows
            a = np.array([ln.split(",")[2:8] for ln in lines if ln], dtype="float64")
        except ValueError:                                  # blanks in a row: let pandas coerce
            m = read_zip_csv(c)
            a = m[MCOLS].apply(pd.to_numeric, errors="coerce").to_numpy("float64")
        if len(a) == 0:
            return (sym, day, 0, *([np.nan] * 12))
        last = pd.DataFrame(a).ffill().iloc[-1].to_numpy()
        return (sym, day, len(a), *last.tolist(), *np.nanmean(a, axis=0).tolist())
    except Exception:                                       # malformed file: skip, keep going
        return (sym, day, -1, *([np.nan] * 12))


def cmd_metrics(shard=0, nshard=1):
    plan = pd.read_csv(os.path.join(OUT, "metrics_plan.csv"), dtype=str)
    out_dir = os.path.join(OUT, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    cols = ["symbol", "day", "rows"] + [f"{c}_last" for c in MCOLS] + [f"{c}_mean" for c in MCOLS]
    for i, (sym, g) in enumerate(plan.groupby("symbol", sort=True)):
        if i % nshard != shard:
            continue
        path = os.path.join(out_dir, f"{sym}.parquet")
        if os.path.exists(path):
            continue
        rows = pmap(lambda d: metrics_day(sym, d), g.day.tolist(), workers=32)
        rows = [r for r in rows if r is not None]
        df = pd.DataFrame(rows, columns=cols).sort_values("day") if rows else pd.DataFrame(columns=cols)
        df.to_parquet(path, index=False)
        print(f"metrics {sym}: {len(df)}/{len(g)} days", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "symbols":
        cmd_symbols()
    elif cmd == "klines":
        cmd_klines(sys.argv[2] if len(sys.argv) > 2 else "1d")
    elif cmd == "funding":
        cmd_funding()
    elif cmd == "metrics":
        cmd_metrics(*(int(x) for x in sys.argv[2:4]))
