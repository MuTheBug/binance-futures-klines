#!/usr/bin/env python3
"""
Download full Binance USDT-M funding-rate history (public, no API key) and resample to daily.

Funding is paid every 8h; we write the daily TOTAL (and mean) per symbol:
    altdata/<BASE>_USDT_funding_1d.csv  ->  date, funding_sum, funding_mean, n

Funding carry was tested before on generic data and rejected OOS, but with real per-symbol
history it can be (a) re-tested as a sleeve against the current ECHO blend and (b) used as a
live cost input. Incremental/resumable like fetch_metrics.py.

Usage:
    python3 fetch_funding.py                  # top-40 liquid crypto bases
    python3 fetch_funding.py --all
    python3 fetch_funding.py --symbols BTC ETH SOL
"""
from __future__ import annotations
import os, glob, time, argparse, datetime as dt
import requests
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"
EXCLUDE = {"NVDA","TSLA","MSTR","SPY","QQQ","INTC","MU","MRVL","AMD","SNDK","SKHYNIX","NOK",
           "EWY","SOXL","CRCL","SPCX","XAG","XAU","XAUT","PAXG","CL","BZ","SLX","BILL","COIN",
           "GENIUS","CHIP","BTW","ZEST"}
SESSION = requests.Session()


def repo_bases():
    bases = []
    for f in glob.glob(os.path.join(ROOT, "*_USDT_1d.csv")):
        b = os.path.basename(f)[:-len("_USDT_1d.csv")]
        if b.isascii() and b not in EXCLUDE:
            bases.append((b, os.path.getsize(f)))
    bases.sort(key=lambda x: -x[1])
    return [b for b, _ in bases]


def fetch_funding(symbol, start_ms):
    """Page forward through funding history from start_ms. Returns list of (ms, rate)."""
    out, cur = [], start_ms
    while True:
        params = {"symbol": symbol, "startTime": cur, "limit": 1000}
        for attempt in range(3):
            try:
                r = SESSION.get(FAPI, params=params, timeout=20)
                if r.status_code == 200:
                    data = r.json(); break
                time.sleep(1 + attempt)
            except requests.RequestException:
                time.sleep(1 + attempt)
        else:
            break
        if not data:
            break
        for d in data:
            out.append((int(d["fundingTime"]), float(d["fundingRate"])))
        last = data[-1]["fundingTime"]
        if len(data) < 1000:
            break
        cur = int(last) + 1
        time.sleep(0.15)
    return out


def process_symbol(base):
    symbol = base + "USDT"
    out_path = os.path.join(OUT, f"{base}_USDT_funding_1d.csv")
    start_ms = 0
    existing = None
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, parse_dates=["date"])
        if len(existing):
            start_ms = int(pd.Timestamp(existing["date"].max()).timestamp() * 1000) + 86_400_000
    rows = fetch_funding(symbol, start_ms)
    if not rows:
        return base, (len(existing) if existing is not None else 0), "up-to-date / none"
    df = pd.DataFrame(rows, columns=["ms", "rate"])
    df["date"] = pd.to_datetime(df["ms"], unit="ms", utc=True).dt.floor("D").dt.tz_localize(None)
    daily = df.groupby("date")["rate"].agg(funding_sum="sum", funding_mean="mean", n="count").reset_index()
    if existing is not None and len(existing):
        daily = pd.concat([existing, daily], ignore_index=True).drop_duplicates("date").sort_values("date")
    daily.to_csv(out_path, index=False)
    return base, len(daily), f"-> {os.path.basename(out_path)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    bases = args.symbols or (repo_bases() if args.all else repo_bases()[:args.top])
    print(f"funding download: {len(bases)} symbols -> altdata/")
    man = []
    for i, base in enumerate(bases):
        try:
            b, n, msg = process_symbol(base)
            print(f"  [{i+1}/{len(bases)}] {b:12} rows={n:<5} {msg}")
            if n:
                man.append({"base": b, "rows": n})
        except Exception as e:
            print(f"  [{i+1}/{len(bases)}] {base:12} ERROR {e}")
    pd.DataFrame(man).to_csv(os.path.join(OUT, "_funding_manifest.csv"), index=False)
    print(f"\nDone. {len(man)} symbols. See altdata/README.md to push.")


if __name__ == "__main__":
    main()
