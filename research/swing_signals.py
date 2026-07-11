"""
Signals for the 4h compression-breakout swing strategy.

Two stages:
  * Base: expensive precomputes done ONCE (ATR, rolling volatility-percentile
    rank, breakout channels for the candidate lookbacks, daily regime/trend
    strength mapped onto the 4h grid with a 1-day information lag, point-in-
    time liquidity eligibility).
  * build(): cheap boolean assembly for a given parameter set — sweeps only
    pay this.

Timing:
  * Daily info for day D (close known at D+1 00:00 UTC) is stamped onto 4h
    bars from D+1 00:00 onwards => usable in signals computed at those bars'
    closes. No look-ahead.
  * Channels use the PRIOR N bars (shift(1)); compression is measured on the
    bar BEFORE the breakout bar.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import swing_data
from swing_engine import Signals

TS_LOOKBACKS = (20, 60)          # daily risk-adjusted momentum lookbacks
TS_K = 1.5                       # tanh squash
EMA_D = 50                       # daily trend EMA
COMP_WINDOW = 252                # 4h bars (~6 weeks) for the vol-percentile rank
CHANNEL_NS = (12, 20, 28, 42)    # breakout lookbacks precomputed for sweeps
MIN_DVOL = 5e6                   # trailing 30d median daily $volume
MIN_HIST_D = 60                  # min daily bars of history


def _wilder_atr(H, L, C, n=14):
    Cp = C.shift(1)
    tr = np.maximum.reduce([(H - L).to_numpy(), (H - Cp).abs().to_numpy(),
                            (L - Cp).abs().to_numpy()])
    tr = pd.DataFrame(tr, index=C.index, columns=C.columns)
    return tr.ewm(alpha=1.0 / n, min_periods=n).mean()


def _daily_to_4h(df_d: pd.DataFrame, idx4: pd.DatetimeIndex, cols) -> pd.DataFrame:
    d = df_d.reindex(columns=cols).copy()
    d.index = d.index + pd.Timedelta(days=1)          # info lag: close of day D known at D+1
    u = d.reindex(d.index.union(idx4)).ffill()
    return u.reindex(idx4)


class Base:
    def __init__(self):
        m4 = swing_data.load_4h()
        md = swing_data.load_daily()
        self.O, self.H, self.L, self.C, self.V = (m4[f] for f in
                                                  ("open", "high", "low", "close", "volume"))
        self.index = self.C.index
        self.symbols = list(self.C.columns)
        cols = self.symbols

        # ---- 4h precomputes ----
        self.atr = _wilder_atr(self.H, self.L, self.C, 14)
        rp = self.atr / self.C                                     # relative vol
        self.rp_rank = rp.rolling(COMP_WINDOW, min_periods=COMP_WINDOW // 2)\
                         .rank(pct=True)                           # percentile of latest
        self.hh = {n: self.H.shift(1).rolling(n, min_periods=n).max() for n in CHANNEL_NS}
        self.ll = {n: self.L.shift(1).rolling(n, min_periods=n).min() for n in CHANNEL_NS}
        med_v = (self.V * self.C).rolling(120, min_periods=30).median()
        self.vol_ratio = (self.V * self.C) / med_v                 # breakout-bar volume conf

        # ---- daily precomputes, mapped to 4h with 1-day lag ----
        Cd, Vd = md["close"].reindex(columns=cols), md["volume"].reindex(columns=cols)
        retd = Cd.pct_change()
        vold = retd.rolling(20, min_periods=15).std()
        ts = sum(np.tanh(TS_K * (Cd / Cd.shift(L) - 1.0) / (vold * np.sqrt(L)))
                 for L in TS_LOOKBACKS) / len(TS_LOOKBACKS)
        ema = Cd.ewm(span=EMA_D, min_periods=EMA_D // 2).mean()
        above = (Cd > ema).astype(float).where(Cd.notna() & ema.notna())

        dvol = (Cd * Vd).rolling(30, min_periods=20).median()
        hist_ok = Cd.notna().cumsum() >= MIN_HIST_D
        elig_d = (dvol >= MIN_DVOL) & hist_ok

        self.ts4 = _daily_to_4h(ts, self.index, cols)
        self.above4 = _daily_to_4h(above, self.index, cols)
        self.elig4 = _daily_to_4h(elig_d.astype(float), self.index, cols).fillna(0.0) > 0.5

        # BTC market regime (gate for new longs), same 1-day info lag
        btc_above = (Cd["BTC"] > ema["BTC"]).astype(float).to_frame("BTC")
        self.mkt4 = _daily_to_4h(btc_above, self.index, ["BTC"])["BTC"].fillna(0.0) > 0.5

        # last valid close per symbol (for delist handling)
        valid = self.C.notna().to_numpy()
        Tn = len(self.index)
        self.last_bar = np.where(valid.any(axis=0),
                                 Tn - 1 - np.argmax(valid[::-1], axis=0), -1)

    def build(self, N: int = 20, comp_q: float = 0.35, ts_thr: float = 0.10,
              vol_conf: float | None = None, require_compression: bool = True,
              mkt_gate: bool = False) -> Signals:
        reg_l = (self.above4 > 0.5) & (self.ts4 > ts_thr)
        reg_s = (self.above4 < 0.5) & (self.ts4 < -ts_thr)

        brk_l = self.C > self.hh[N]
        brk_s = self.C < self.ll[N]
        ok = self.elig4 & self.atr.notna() & self.C.notna()
        if require_compression:
            comp_prev = self.rp_rank.shift(1) <= comp_q
            ok = ok & comp_prev
        if vol_conf is not None:
            ok = ok & (self.vol_ratio >= vol_conf)

        e_l = (brk_l & reg_l & ok).fillna(False)
        e_s = (brk_s & reg_s & ok).fillna(False)
        if mkt_gate:                     # block NEW longs when BTC below daily EMA50
            e_l = e_l & self.mkt4.to_numpy()[:, None]
            e_s = e_s & (~self.mkt4.to_numpy())[:, None]

        return Signals(
            index=self.index, symbols=self.symbols,
            O=self.O.to_numpy("float64"), H=self.H.to_numpy("float64"),
            L=self.L.to_numpy("float64"), C=self.C.to_numpy("float64"),
            entry_long=e_l.to_numpy(bool), entry_short=e_s.to_numpy(bool),
            regime_long=reg_l.fillna(False).to_numpy(bool),
            regime_short=reg_s.fillna(False).to_numpy(bool),
            conviction=self.ts4.abs().fillna(0.0).to_numpy("float64"),
            atr=self.atr.to_numpy("float64"),
            level_long=self.hh[N].to_numpy("float64"),
            level_short=self.ll[N].to_numpy("float64"),
            last_bar=self.last_bar,
        )
