"""Market data sources.

BinanceData : live public Binance data (no API key needed) - universe, daily klines, mark prices,
              funding. Only CLOSED daily bars are returned.
CsvData     : offline source over this repo's <BASE>_USDT_1d.csv files (replay and tests).
"""
from __future__ import annotations
import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from .binance import BinanceFuturesClient, SymbolFilters, parse_filters

log = logging.getLogger(__name__)
DAY_MS = 86_400_000


@dataclass
class Bars:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def last_day(self) -> pd.Timestamp:
        return self.close.index[-1]


def last_closed_day(now: pd.Timestamp) -> pd.Timestamp:
    """The most recent daily bar whose close (00:00 UTC next day) is <= now."""
    return now.tz_convert("UTC").floor("D") - pd.Timedelta(days=1)


def _frames(rows: dict[str, list], end_day: pd.Timestamp, n: int) -> Bars:
    grid = pd.date_range(end=end_day, periods=n, freq="D", tz="UTC")
    out = {}
    for i, name in enumerate(("open", "high", "low", "close", "volume")):
        cols = {}
        for sym, kl in rows.items():
            if not kl:
                continue
            idx = pd.to_datetime([k[0] for k in kl], unit="ms", utc=True)
            cols[sym] = pd.Series([float(k[i + 1]) for k in kl], index=idx)
        df = pd.DataFrame(cols).reindex(grid)
        out[name] = df
    return Bars(**out)


class BinanceData:
    def __init__(self, client: BinanceFuturesClient, exclude=(), min_quote_volume_24h=1e6, max_workers=8):
        self.client = client
        self.exclude = list(exclude)
        self.min_qv = min_quote_volume_24h
        self.max_workers = max_workers
        self._filters: dict[str, SymbolFilters] | None = None

    def filters(self, refresh=False) -> dict[str, SymbolFilters]:
        if self._filters is None or refresh:
            self._filters = parse_filters(self.client.exchange_info(), self.exclude)
        return self._filters

    def universe(self, must_include=()) -> list[str]:
        f = self.filters(refresh=True)
        qv = {t["symbol"]: float(t.get("quoteVolume", 0)) for t in self.client.ticker_24h()}
        syms = sorted(s for s in f if qv.get(s, 0.0) >= self.min_qv)
        for s in must_include:
            if s in f and s not in syms:
                syms.append(s)
        log.info("universe: %d tradable USDT perps, %d pass 24h volume prefilter", len(f), len(syms))
        return syms

    def daily_bars(self, symbols, n: int, now: pd.Timestamp) -> Bars:
        end_day = last_closed_day(now)
        now_ms = int(now.timestamp() * 1000)

        def fetch(sym):
            kl = self.client.klines(sym, "1d", limit=min(n + 2, 1500))
            return sym, [k for k in kl if int(k[6]) < now_ms]      # keep closed bars only

        with ThreadPoolExecutor(self.max_workers) as ex:
            rows = dict(ex.map(fetch, symbols))
        return _frames(rows, end_day, n)

    def mark_prices(self) -> dict[str, float]:
        return {p["symbol"]: float(p["markPrice"]) for p in self.client.premium_index()}

    def funding_paid(self, symbols, start_ms: int, end_ms: int) -> dict[str, float]:
        """Sum of funding rates per symbol in (start, end]. Positive = longs paid."""
        out = {}
        for s in symbols:
            rows = self.client.funding_rates(s, start_ms + 1, end_ms)
            out[s] = sum(float(r["fundingRate"]) for r in rows)
        return out


class CsvData:
    """Offline daily data from the repo CSVs. Symbols are exposed as <BASE>USDT."""

    def __init__(self, csv_dir: str, exclude=()):
        self.exclude = set(exclude)
        self._load(csv_dir)

    def _load(self, csv_dir):
        cols = {k: {} for k in ("open", "high", "low", "close", "volume")}
        for f in glob.glob(os.path.join(csv_dir, "*_USDT_1d.csv")):
            base = os.path.basename(f)[: -len("_USDT_1d.csv")]
            if base in self.exclude:
                continue
            df = pd.read_csv(f).drop_duplicates("timestamp")
            idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for k in cols:
                cols[k][base + "USDT"] = pd.Series(df[k].to_numpy(float), index=idx)
        if not cols["close"]:
            raise FileNotFoundError(f"no *_USDT_1d.csv files in {csv_dir}")
        full = {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}
        grid = pd.date_range(full["close"].index[0], full["close"].index[-1], freq="D")
        self.all = Bars(**{k: v.reindex(grid) for k, v in full.items()})

    def filters(self, refresh=False) -> dict[str, SymbolFilters]:
        return {s: SymbolFilters(s, s[:-4], Decimal("0.00000001"), Decimal("0.00000001"),
                                 Decimal("1e15"), Decimal("0.00000001"), 5.0)
                for s in self.all.close.columns}

    def universe(self, must_include=()) -> list[str]:
        return list(self.all.close.columns)

    def daily_bars(self, symbols, n: int, now: pd.Timestamp) -> Bars:
        end = last_closed_day(now)
        sl = lambda d: d.loc[:end, list(symbols)].iloc[-n:]
        b = Bars(*(sl(getattr(self.all, k)) for k in ("open", "high", "low", "close", "volume")))
        return b

    def mark_prices(self, day: pd.Timestamp | None = None) -> dict[str, float]:
        row = self.all.close.loc[day] if day is not None else self.all.close.ffill().iloc[-1]
        return row.dropna().to_dict()
