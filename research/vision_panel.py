"""Research panel built from data.binance.vision downloads (see fetch_vision.py, V5_PROTOCOL.md).

BLINDNESS: every loader cuts at DESIGN_END unless fresh=True. Only fresh_test.py may pass
fresh=True, once, after the candidates are frozen and committed.

Columns are base names: BTCUSDT -> BTC. Old histories of contracts that were delisted and later
relisted under the same name (XXXUSDTSETTLED) are separate instruments: XXX.S1 (XXX.S2 ...).
"""
from __future__ import annotations
import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIS = os.path.join(ROOT, "data", "vision")
PANEL = os.path.join(VIS, "panel")
DESIGN_END = pd.Timestamp("2026-06-07", tz="UTC")
IS_END = pd.Timestamp("2024-12-31", tz="UTC")
PANEL_START = pd.Timestamp("2020-01-01", tz="UTC")

# ---------------------------------------------------------------------------------------------
# Not crypto, decided from SYMBOL NAMES ONLY (2026-11-10, before any price was looked at).
# Tokenized stocks / ETFs / leveraged ETFs / commodities / FX / pre-IPO products:
_TRADFI = """
AAOI AAPL ABBV ACN ADBE AGPU AKAM ALAB AMAT AMC AMD AMGN AMZN ANET ANTHROPIC APLD APP ARM ASML ASTS
AVGO AXTI BABA BAC BE BILL BITO BMNR BRKB BWET BX BYD BZ CAT CBRS CEG CHIP CIEN CL COHR COIN COPPER
COST CRCL CRDO CRML CRM CRWD CRWV CSCO CSOPSAMSUNG2L CSOPSKHYNIX2L CVNA CXMT CYPH DDOG DELL DIS DJT
DKNG EBAY EIDO EQT EWJ EWT EWY EWZS EWZ FLEX FLNC FWDI GDX GENIUS GEV GIGADEV GLW GME GOOGL GPRO GS
GTLB HANMI HD HIMS HK0625 HK0700 HK0992 HK1810 HOOD HPE HUT HYUNDAI IBKR IBM INTC INTW IONQ IREN IWM
JPM KLAC KODEX200 KORU KO KSTR KUAISHOU LGELECTRONICS LITE LLY LRCX MARA MCD MDB MEITUAN MELI META
MINIMAX MOONSHOT MP MRK MRNA MRVL MSFT MSTR MSTU MU MUU MVLL NATGAS NAVER NBIS NET NFLX NKE NOK NOW
NVDA NVDL NVO OKLO ONDS OPENAI ORCL OURA OXY PANW PATH PAXG PAYP PDD PFE PLTR POPMART PYPL QCOM QNTX
QQQ RDDT RDW RIVN RKLB RUM SAMSUNGEM SAMSUNG SHOP SKHYNIX SKHY SLX SMCI SMH SMR SNDK SNOW SNPS SNXX
SOFI SONY SOXL SOXS SPCX SPY SQQQ STM STRC STXX TBT TEAM TEM TENCENT TER TMF TQQQ TSLA TSLL TSM TTWO
TWST TXN TZA UBER UNH UNITREE URNM USAR USDBRL UVXY VKTX VRT VST V WDC WMT XAG XAUT XAU XBI XLE XOM
XPD XPT ZCSH ZHIPU ZHONGJI ZM ZS
"""
# Index products and stablecoin pairs:
_INDEX_STABLE = "BTCDOM DEFI BLUEBIRD FOOTBALL DOTECO USDC"
NON_CRYPTO = set(_TRADFI.split()) | set(_INDEX_STABLE.split())
# Names that could be either (same ticker as a stock and a token, or unknown to the author);
# left to the data-driven filter below. Listed for the classifier validation report only.
AMBIGUOUS = set("""ALL BNC DRAM FIS F ON OPEN O RE RAM ROBO STAR SPACE USDEX GRAM LYTE PENG PONS PRL
OPG OPN BSB BSP BTW CT CTR DOS SECZ SHAZ SKDD SKUU ARX BBX DATAIP INX KAT 42 US ZEST MARSCOIN ELSA
CAP BOT ACU""".split())

FIELDS = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote_volume", "count"]


def base_name(sym: str) -> str:
    m = re.match(r"^(.+?)USDT((?:_?SETTLED)*)$", sym)
    n_settled = len(re.findall("SETTLED", m.group(2)))
    return m.group(1) + (f".S{n_settled}" if n_settled else "")


def is_crypto_name(col: str) -> bool:
    return col.split(".")[0] not in NON_CRYPTO


# --------------------------------------------------------------------------------------- build
def build_panel():
    """Parse per-symbol parquet files into aligned daily matrices (full date range) and cache."""
    os.makedirs(PANEL, exist_ok=True)
    frames = {f: {} for f in FIELDS}
    for path in sorted(glob.glob(os.path.join(VIS, "klines_1d", "*.parquet"))):
        k = pd.read_parquet(path)
        if len(k) == 0:
            continue
        idx = pd.to_datetime(k["open_time"], unit="ms", utc=True)
        col = base_name(os.path.basename(path)[:-8])
        for f in FIELDS:
            frames[f][col] = pd.Series(k[f].to_numpy("float64"), index=idx)
    first = min(s.index.min() for s in frames["close"].values())
    last = max(s.index.max() for s in frames["close"].values())
    grid = pd.date_range(max(first, PANEL_START), last, freq="D")
    traded = pd.DataFrame(frames["volume"]).reindex(grid) > 0
    mats = {f: pd.DataFrame(frames[f]).reindex(grid).where(traded) for f in FIELDS}
    n_fixed = apply_repairs(mats)
    for f in FIELDS:
        mats[f].to_parquet(os.path.join(PANEL, f"{f}.parquet"))
    if n_fixed:
        print(f"archive-hole repair: {n_fixed} symbol-days rebuilt from mark-price bars")
    return grid, len(frames["close"])


def apply_repairs(mats: dict) -> int:
    """Rebuild days the kline archive lacks from Binance's daily mark-price bars
    (data/vision/repair_markprice_<day>.parquet; see V5_PROTOCOL.md D2). Only fills a symbol whose
    bar is missing that day and present the day before. Volume fields = mean of adjacent days."""
    n = 0
    for path in sorted(glob.glob(os.path.join(VIS, "repair_markprice_*.parquet"))):
        day = pd.Timestamp(os.path.basename(path)[len("repair_markprice_"):-8], tz="UTC")
        if day not in mats["close"].index:
            continue
        mp = pd.read_parquet(path)
        mp.index = [base_name(x) for x in mp.symbol]
        prev, nxt = day - pd.Timedelta(days=1), day + pd.Timedelta(days=1)
        c = mats["close"]
        cols = [k for k in mp.index if k in c.columns and pd.isna(c.at[day, k]) and pd.notna(c.at[prev, k])]
        for f in ("open", "high", "low", "close"):
            mats[f].loc[day, cols] = mp.loc[cols, f].to_numpy("float64")
        for f in ("volume", "quote_volume", "taker_buy_quote_volume", "count"):
            adj = mats[f].loc[[prev, nxt], cols] if nxt in mats[f].index else mats[f].loc[[prev], cols]
            mats[f].loc[day, cols] = adj.mean(axis=0).to_numpy("float64")
        n += len(cols)
    return n


def build_funding(grid):
    """Daily funding per symbol aligned to HOLDING days: an event at time T belongs to the UTC day
    containing T - 1h (so the 00:00 payment closes the previous day's hold)."""
    cols = {}
    for path in sorted(glob.glob(os.path.join(VIS, "funding", "*.parquet"))):
        f = pd.read_parquet(path)
        if len(f) == 0:
            continue
        t = pd.to_datetime(pd.to_numeric(f["calc_time"]), unit="ms", utc=True) - pd.Timedelta(hours=1)
        cols[base_name(os.path.basename(path)[:-8])] = (
            pd.Series(pd.to_numeric(f["last_funding_rate"]).to_numpy("float64"), index=t.dt.floor("D")).groupby(level=0).sum())
    pd.DataFrame(cols).reindex(grid).to_parquet(os.path.join(PANEL, "funding.parquet"))


def build_metrics(grid):
    names = {"sum_open_interest_last": "oi", "sum_open_interest_value_last": "oiv",
             "count_long_short_ratio_mean": "lsa", "sum_toptrader_long_short_ratio_mean": "ttp",
             "count_toptrader_long_short_ratio_mean": "tta", "sum_taker_long_short_vol_ratio_mean": "taker"}
    out = {v: {} for v in names.values()}
    for path in sorted(glob.glob(os.path.join(VIS, "metrics", "*.parquet"))):
        m = pd.read_parquet(path)
        m = m[m["rows"] > 0]
        if len(m) == 0:
            continue
        idx = pd.to_datetime(m["day"], utc=True)
        col = base_name(os.path.basename(path)[:-8])
        for src, dst in names.items():
            out[dst][col] = pd.Series(pd.to_numeric(m[src], errors="coerce").to_numpy("float64"), index=idx)
    for dst, cols in out.items():
        pd.DataFrame(cols).reindex(grid).to_parquet(os.path.join(PANEL, f"m_{dst}.parquet"))


# ---------------------------------------------------------------------------------------- load
def load(field: str, fresh: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(PANEL, f"{field}.parquet"))
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC") if df.index.tz is not None else pd.DatetimeIndex(df.index, tz="UTC")
    return df if fresh else df.loc[:DESIGN_END]


def weekend_ratio(ret: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Robust weekend/weekday activity: median |return| on Sat+Sun bars divided by the median
    |return| on Mon-Fri bars over the trailing window. Markets that close on weekends score ~0."""
    a = ret.abs()
    wk = pd.Series(ret.index.dayofweek >= 5, index=ret.index)
    we = a.where(wk, axis=0).rolling(window, min_periods=8).median()       # >= 4 weekends
    wd = a.where(~wk, axis=0).rolling(window, min_periods=20).median()     # >= 4 working weeks
    return we / wd.replace(0, np.nan)


NEW_LISTING_CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")   # tokenized TradFi perps did not exist before


def crypto_mask(close: pd.DataFrame, weekend_thr: float, vol_floor: float = 0.005,
                full_close: pd.DataFrame | None = None) -> pd.DataFrame:
    """Point-in-time 'is a crypto coin' mask: on the name list, not stablecoin-like, and (only for
    instruments first listed in 2025 or later) not weekend-dead. The weekend rule is not applied
    to older listings: BTC/ETH/SOL have quiet weekends too and must never be dropped."""
    ret = close.pct_change(fill_method=None)
    by_name = pd.Series([is_crypto_name(c) for c in close.columns], index=close.columns)
    first = (full_close if full_close is not None else close).apply(lambda s: s.first_valid_index())
    is_new = (first >= NEW_LISTING_CUTOFF).fillna(False).to_numpy()
    wr_ok = (weekend_ratio(ret) >= weekend_thr) | ~is_new[None, :]
    vol = ret.rolling(30, min_periods=15).std()
    return wr_ok & (vol >= vol_floor) & by_name.to_numpy()[None, :]


def eligibility(close, quote_volume, crypto, min_history=60, min_qv=5e6, liq_lb=30):
    has = close.notna()
    hist_ok = has.rolling(min_history, min_periods=min_history).sum() >= min_history
    liq = quote_volume.rolling(liq_lb, min_periods=max(5, liq_lb // 2)).median()
    return has & hist_ok & (liq >= min_qv) & crypto


def write_metrics_plan(weekend_thr: float, warmup_days: int = 35):
    """Symbol-days to download positioning metrics for: from 35 days before an instrument first
    becomes eligible to its last eligible day. Mechanical; uses the full date range."""
    c, qv = load("close", fresh=True), load("quote_volume", fresh=True)
    e = eligibility(c, qv, crypto_mask(c, weekend_thr))
    files = {base_name(os.path.basename(p)[:-8]): os.path.basename(p)[:-8]
             for p in glob.glob(os.path.join(VIS, "klines_1d", "*.parquet"))}
    rows = []
    for col in e.columns[e.any()]:
        days = e.index[e[col]]
        lo = max(days[0] - pd.Timedelta(days=warmup_days), pd.Timestamp("2020-09-01", tz="UTC"))
        for d in pd.date_range(lo, days[-1], freq="D"):
            rows.append((files[col], str(d.date())))
    pd.DataFrame(rows, columns=["symbol", "day"]).to_csv(os.path.join(VIS, "metrics_plan.csv"), index=False)
    return len(rows), int(e.any().sum())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "plan":
        from v5_lib import WEEKEND_THR
        n, k = write_metrics_plan(WEEKEND_THR)
        print(f"metrics plan: {n} symbol-days for {k} instruments")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "metrics":
        g = load("close", fresh=True).index
        build_metrics(g); print("metrics matrices built")
        sys.exit(0)
    grid, n = build_panel()
    print(f"panel: {n} instruments, {grid[0].date()} .. (last date not printed: blind)")
    if glob.glob(os.path.join(VIS, "funding", "*.parquet")):
        build_funding(grid); print("funding matrix built")
    if glob.glob(os.path.join(VIS, "metrics", "*.parquet")):
        build_metrics(grid); print("metrics matrices built")
