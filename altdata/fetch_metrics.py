#!/usr/bin/env python3
"""
Download Binance USDT-M futures *metrics* (the orthogonal data that breaks the OHLCV ceiling)
from data.binance.vision — multi-year history, no API key, no 30-day limit.

Per symbol per day Binance publishes a zip of 5-minute metrics rows with columns:
    create_time, symbol,
    sum_open_interest, sum_open_interest_value,                 # open interest (coin / USDT)
    count_toptrader_long_short_ratio,                           # top traders, by ACCOUNT
    sum_toptrader_long_short_ratio,                             # top traders, by POSITION
    count_long_short_ratio,                                     # GLOBAL accounts (retail)
    sum_taker_long_short_vol_ratio                              # aggressive taker buy/sell flow

We resample to DAILY (mean over the UTC day) and write one tidy CSV per symbol:
    altdata/<BASE>_USDT_metrics_1d.csv
      date, oi, oi_value, toptrader_acc_ls, toptrader_pos_ls, global_acc_ls, taker_ls

Why these matter (each ~orthogonal to price-trend):
  * d(open interest) vs price  -> new money vs short-covering (positioning regime)
  * top-trader long/short      -> "smart money" lean (follow)
  * global account long/short  -> retail lean (often contrarian -> fade)
  * taker buy/sell ratio       -> aggressive order-flow imbalance

Incremental & resumable: re-running only fetches days after the last row already saved
(so it doubles as a daily collector via cron). Use --symbols / --top / --start to scope.

Usage:
    python3 fetch_metrics.py                 # default: top-40 liquid crypto bases, full history
    python3 fetch_metrics.py --all           # every crypto base present in the repo (large!)
    python3 fetch_metrics.py --symbols BTC ETH SOL --start 2022-01-01
"""
from __future__ import annotations
import os, sys, io, zipfile, glob, time, argparse, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
OUT = os.path.dirname(os.path.abspath(__file__))                     # altdata/
BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
DEFAULT_START = "2021-01-01"
# non-crypto (tokenized stocks/metals/FX) — skip
EXCLUDE = {"NVDA","TSLA","MSTR","SPY","QQQ","INTC","MU","MRVL","AMD","SNDK","SKHYNIX","NOK",
           "EWY","SOXL","CRCL","SPCX","XAG","XAU","XAUT","PAXG","CL","BZ","SLX","BILL","COIN",
           "GENIUS","CHIP","BTW","ZEST"}
COLS = {  # vision column -> our name
    "sum_open_interest": "oi", "sum_open_interest_value": "oi_value",
    "count_toptrader_long_short_ratio": "toptrader_acc_ls",
    "sum_toptrader_long_short_ratio": "toptrader_pos_ls",
    "count_long_short_ratio": "global_acc_ls",
    "sum_taker_long_short_vol_ratio": "taker_ls",
}
SESSION = requests.Session()


def repo_bases():
    bases = []
    for f in glob.glob(os.path.join(ROOT, "*_USDT_1d.csv")):
        b = os.path.basename(f)[:-len("_USDT_1d.csv")]
        if b.isascii() and b not in EXCLUDE:
            bases.append((b, os.path.getsize(f)))
    bases.sort(key=lambda x: -x[1])               # bigger 1d file ~ longer history / liquidity
    return [b for b, _ in bases]


def fetch_day(symbol, day):
    """Return a DataFrame of one day's 5-min metrics rows, or None if absent."""
    url = f"{BASE_URL}/{symbol}/{symbol}-metrics-{day}.zip"
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(1 + attempt); continue
        if r.status_code == 404:
            return None
        if r.status_code == 200:
            try:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                df = pd.read_csv(z.open(z.namelist()[0]))
                return df
            except Exception:
                return None
        time.sleep(1 + attempt)
    return None


def daterange(start, end):
    d = start
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += dt.timedelta(days=1)


def process_symbol(base, start_date, end_date, threads=12):
    symbol = base + "USDT"
    out_path = os.path.join(OUT, f"{base}_USDT_metrics_1d.csv")
    existing = None
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, parse_dates=["date"])
        if len(existing):
            last = existing["date"].max().date()
            start_date = max(start_date, last + dt.timedelta(days=1))
    days = list(daterange(start_date, end_date))
    if not days:
        return base, 0, "up-to-date"
    frames = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = {ex.submit(fetch_day, symbol, d): d for d in days}
        for fu in as_completed(futs):
            df = fu.result()
            if df is not None and len(df):
                frames.append(df)
    if not frames:
        return base, 0, "no data (symbol may lack metrics)"
    raw = pd.concat(frames, ignore_index=True)
    tcol = "create_time" if "create_time" in raw.columns else raw.columns[0]
    raw["date"] = pd.to_datetime(raw[tcol], utc=True).dt.floor("D")
    keep = {v: k for k, v in COLS.items() if k in raw.columns}
    daily = raw.groupby("date")[list(keep.values())].mean().rename(columns={v: k for k, v in keep.items()})
    daily = daily.reset_index()
    daily["date"] = daily["date"].dt.tz_localize(None)
    if existing is not None and len(existing):
        daily = pd.concat([existing, daily], ignore_index=True).drop_duplicates("date").sort_values("date")
    daily.to_csv(out_path, index=False)
    return base, len(daily), f"-> {os.path.basename(out_path)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", help="explicit base list, e.g. BTC ETH SOL")
    ap.add_argument("--all", action="store_true", help="all crypto bases in the repo")
    ap.add_argument("--top", type=int, default=40, help="top-N liquid bases (default 40)")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    if args.symbols:
        bases = args.symbols
    else:
        bases = repo_bases()
        if not args.all:
            bases = bases[:args.top]
    start_date = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = (dt.datetime.utcnow() - dt.timedelta(days=1)).date()
    print(f"metrics download: {len(bases)} symbols, {start_date}..{end_date} -> altdata/")
    man = []
    for i, base in enumerate(bases):
        try:
            b, n, msg = process_symbol(base, start_date, end_date, args.threads)
            print(f"  [{i+1}/{len(bases)}] {b:12} rows={n:<5} {msg}")
            if n:
                man.append({"base": b, "rows": n})
        except Exception as e:
            print(f"  [{i+1}/{len(bases)}] {base:12} ERROR {e}")
        time.sleep(0.1)
    pd.DataFrame(man).to_csv(os.path.join(OUT, "_metrics_manifest.csv"), index=False)
    print(f"\nDone. {len(man)} symbols written. Next: see altdata/README.md to push.")


if __name__ == "__main__":
    main()
