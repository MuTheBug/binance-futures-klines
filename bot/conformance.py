"""Conformance audit: does the bot trade the backtested strategy?

Compares, for the same as-of date:
  T_bot      targets from bot/signals.compute_targets via an offline
             exchange serving the repo's own CSVs + committed alt-data
             (the bot's real code path, real config)
  T_research targets from the research pipeline on FULL history and FULL
             universe (what the backtest holds on that date)

and attributes any gap to its two known input differences:
  (1) the bot fetches only the last 300 daily bars
  (2) the bot excludes tokenized stocks/metals (EXCLUDE list)

Run:  cd bot && python3 conformance.py
"""
from __future__ import annotations
import glob
import os
import sys

import numpy as np
import pandas as pd

from config import Config
from selftest import FakeExchange
import signals


class FakeExchangeWithPositioning(FakeExchange):
    """Also serves the committed alt-data so sleeve C is active offline."""
    def top_long_short_position_ratio(self, symbol, period="1d", limit=30):
        base = symbol[:-4]
        f = os.path.join(self.root, "altdata", f"{base}_USDT_metrics_1d.csv")
        if not os.path.exists(f):
            return []
        df = pd.read_csv(f, usecols=["date", "toptrader_pos_ls"]).tail(limit)
        out = []
        for _, r in df.iterrows():
            ts = int(pd.Timestamp(r["date"], tz="UTC").value // 1_000_000)
            out.append((ts, float(r["toptrader_pos_ls"])))
        return out


def research_targets(cfg, st, engine, echo, altdata, lab,
                     restrict_hist=None, exclude=None):
    close, vol, ret, elig = lab.load_data("1d")
    if exclude:
        keep = [c for c in close.columns if c not in exclude]
        close, vol, ret, elig = close[keep], vol[keep], ret[keep], elig[keep]
        elig = st.eligibility(close, vol, cfg.min_history_days,
                              cfg.min_dollar_vol, 30)
    if restrict_hist:
        close, vol = close.tail(restrict_hist), vol.tail(restrict_hist)
        ret = engine.to_returns(close)
        elig = st.eligibility(close, vol, cfg.min_history_days,
                              cfg.min_dollar_vol, 30)
    sigT = st.strength_signal(close, lookbacks=(15, 30, 60, 90),
                              strength_vol=30, k=2.0)
    wA = st.concentrate(st.ema_ensemble(
        st.signal_to_weights(sigT, ret, elig, 15), (5, 10, 15)), 10)
    wB = echo.residual_continuation(close, ret, elig, beta_window=60,
                                    formations=(3, 5, 8), cap=0.10)
    ls = altdata.load_metric("toptrader_pos_ls", reindex_like=close)
    wC = echo.positioning_sleeve(close, ret, elig, ls, fade=True, lag=1,
                                 beta_window=60, cap=0.10)
    blend = 0.40 * wA + 0.40 * wB + 0.20 * wC
    wv = engine.vol_target(blend, ret, "1d", target_vol=cfg.target_vol,
                           max_leverage=3.0)
    last = wv.iloc[-1].dropna()
    last = last[last.abs() > 1e-6] * cfg.leverage
    if cfg.conviction_floor > 0 and len(last) > 5:
        thr = last.abs().quantile(cfg.conviction_floor)
        g0 = float(last.abs().sum())
        last = last[last.abs() >= thr]
        if last.abs().sum() > 0:
            last = last * (g0 / float(last.abs().sum()))
    if len(last) > cfg.max_positions:
        last = last.reindex(last.abs().sort_values(ascending=False)
                            .index[:cfg.max_positions])
    gross = float(last.abs().sum())
    if gross > cfg.max_gross:
        last = last * (cfg.max_gross / gross)
        gross = cfg.max_gross
    if 0 < gross < cfg.min_gross:
        last = last.iloc[0:0]
    return {b + "USDT": float(w) for b, w in last.items()}, close.index[-1]


def compare(a, b, la, lb):
    syms = sorted(set(a) | set(b))
    va = np.array([a.get(s, 0.0) for s in syms])
    vb = np.array([b.get(s, 0.0) for s in syms])
    ga = np.abs(va).sum()
    overlap = 1.0 - np.abs(va - vb).sum() / (np.abs(va).sum() + np.abs(vb).sum())
    corr = np.corrcoef(va, vb)[0, 1] if len(syms) > 2 else np.nan
    print(f"  {la} vs {lb}: weight-corr {corr:+.3f}   "
          f"portfolio-overlap {overlap * 100:.1f}%   "
          f"gross {np.abs(va).sum():.2f}x/{np.abs(vb).sum():.2f}x")
    diffs = sorted(syms, key=lambda s: -abs(a.get(s, 0) - b.get(s, 0)))[:5]
    for s in diffs:
        print(f"     {s:<14} {la}: {a.get(s, 0) * 100:+6.2f}%   "
              f"{lb}: {b.get(s, 0) * 100:+6.2f}%")
    return overlap


def main():
    cfg = Config.load()
    cfg.blend_echo = 1.0          # audit against the APEX (pure-ECHO) backtest
    root = os.path.dirname(cfg.research_dir)
    sys.path.insert(0, cfg.research_dir)
    import strategies as st
    import engine, echo, altdata, lab

    ex = FakeExchangeWithPositioning(root)
    t_bot, info = signals.compute_targets(ex, cfg, log=lambda *a: None)
    print(f"BOT targets   as-of {info['asof']}  sleeves={info['sleeves']}  "
          f"n={info['n_targets']}  gross={info['gross']}x  net={info['net']}")

    t_res, asof = research_targets(cfg, st, engine, echo, altdata, lab)
    print(f"RESEARCH targets as-of {asof.date()}  n={len(t_res)}")
    print("\n[1] bot vs research (full history, full universe):")
    compare(t_bot, t_res, "bot", "res")

    ex_set = signals.EXCLUDE
    t_res_ex, _ = research_targets(cfg, st, engine, echo, altdata, lab,
                                   exclude=ex_set)
    print("\n[2] bot vs research + bot's EXCLUDE list:")
    compare(t_bot, t_res_ex, "bot", "res")

    t_res_both, _ = research_targets(cfg, st, engine, echo, altdata, lab,
                                     restrict_hist=cfg.klines_limit,
                                     exclude=ex_set)
    print("\n[3] bot vs research + EXCLUDE + 300-day history (bot's exact inputs):")
    ov = compare(t_bot, t_res_both, "bot", "res")
    print(f"\nVERDICT: {'CONFORMS' if ov > 0.9 else 'DIVERGES -- investigate'} "
          f"(residual overlap {ov * 100:.1f}%)")


if __name__ == "__main__":
    main()
