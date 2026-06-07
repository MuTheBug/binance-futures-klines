"""Fetch full funding-rate history from Binance for every crypto symbol we have klines for.
Saves data/funding/<BASE>_USDT_funding.csv (fundingTime, fundingRate, markPrice).
Resumable: skips symbols already saved. Parallel (threads) + paginated."""
from __future__ import annotations
import os, re, glob, json, time
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "funding")
os.makedirs(OUT, exist_ok=True)
START = 1546300800000           # 2019-01-01
URL = "https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&startTime={st}&limit=1000"


def bases():
    out = []
    for f in glob.glob(os.path.join(ROOT, "*_USDT_1d.csv")):
        b = os.path.basename(f)[: -len("_USDT_1d.csv")]
        if re.fullmatch(r"[A-Z0-9]+", b):     # skip non-ascii / odd names
            out.append(b)
    return sorted(set(out))


def fetch_one(base):
    sym = base + "USDT"
    path = os.path.join(OUT, f"{base}_USDT_funding.csv")
    if os.path.exists(path) and os.path.getsize(path) > 100:
        return base, -1, "skip(exists)"
    rows, st = [], START
    for _ in range(60):
        try:
            req = urllib.request.Request(URL.format(sym=sym, st=st), headers={"User-Agent": "research"})
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            return base, None, f"http{e.code}"
        except Exception as e:
            return base, None, type(e).__name__
        if not data:
            break
        rows += [(d["fundingTime"], d["fundingRate"], d.get("markPrice")) for d in data]
        if len(data) < 1000:
            break
        st = data[-1]["fundingTime"] + 1
        time.sleep(0.03)
    if not rows:
        return base, 0, "empty"
    df = pd.DataFrame(rows, columns=["fundingTime", "fundingRate", "markPrice"]).drop_duplicates("fundingTime")
    df.to_csv(path, index=False)
    return base, len(df), "ok"


def main():
    bs = bases()
    print(f"fetching funding for {len(bs)} symbols -> {OUT}")
    ok = fail = skip = 0
    fails = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_one, b): b for b in bs}
        for i, fut in enumerate(as_completed(futs)):
            base, n, status = fut.result()
            if status == "ok":
                ok += 1
            elif status.startswith("skip"):
                skip += 1
            else:
                fail += 1; fails.append(f"{base}:{status}")
            if i % 25 == 0:
                print(f"  {i}/{len(bs)} done (ok={ok} skip={skip} fail={fail})")
    print(f"\nDONE: ok={ok} skip={skip} fail={fail}")
    if fails:
        print("failures:", ", ".join(fails[:40]))
    # quick coverage summary
    files = glob.glob(os.path.join(OUT, "*_funding.csv"))
    print(f"total funding files: {len(files)}")


if __name__ == "__main__":
    main()
