"""Build an aligned hourly panel (opens/closes/dollar-volume) from the raw kline CSVs.

Point-in-time hygiene:
- every symbol keeps its own listing date; nothing is backfilled
- the tradable universe at hour t uses only data available strictly before t
  (rolling 30d median hourly dollar volume, shifted by 1 bar)
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = "/tmp/claude-0/-home-user-binance-futures-klines/b68190b2-2a75-5b5b-b433-5f5a3e931263/scratchpad/cache/panel.pkl"

LIQ_WINDOW = 24 * 30          # 30 days of hourly bars
LIQ_MIN_PERIODS = 24 * 21     # need >= 3 weeks of history before a coin is tradable
LIQ_FLOOR_USD = 1_000_000     # median hourly dollar volume >= $1M


def build_panel(force: bool = False):
    if os.path.exists(CACHE) and not force:
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)

    opens, closes, dvols = {}, {}, {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_USDT_1h.csv"))):
        sym = os.path.basename(path).replace("_USDT_1h.csv", "")
        df = pd.read_csv(path, usecols=["timestamp", "open", "close", "volume"])
        if len(df) < LIQ_MIN_PERIODS:
            continue
        idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index(idx).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        opens[sym] = df["open"]
        closes[sym] = df["close"]
        dvols[sym] = df["close"] * df["volume"]

    O = pd.DataFrame(opens)
    C = pd.DataFrame(closes)
    DV = pd.DataFrame(dvols)
    full_idx = pd.date_range(C.index.min(), C.index.max(), freq="1h", tz="UTC")
    O, C, DV = (x.reindex(full_idx) for x in (O, C, DV))

    liq = DV.rolling(LIQ_WINDOW, min_periods=LIQ_MIN_PERIODS).median()
    universe = (liq.shift(1) >= LIQ_FLOOR_USD) & C.notna()

    panel = {"O": O, "C": C, "DV": DV, "universe": universe}
    with open(CACHE, "wb") as fh:
        pickle.dump(panel, fh)
    return panel


if __name__ == "__main__":
    p = build_panel(force=True)
    C, U = p["C"], p["universe"]
    print("bars:", len(C), "symbols:", C.shape[1])
    print("span:", C.index[0], "->", C.index[-1])
    print("universe size over time (mean/min/max):",
          round(U.sum(axis=1).mean(), 1), int(U.sum(axis=1).min()), int(U.sum(axis=1).max()))
