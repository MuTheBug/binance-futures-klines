"""
DEFINITIVE swing strategy runner — "CBS v2": channel-breakout swing, long-only,
retest limit entries, close-based chandelier stops.

Frozen config (tuned ONLY on data <= 2024-12-31; see sweep_swing.py and
SWING_STRATEGY.md for the search history and the plateau evidence):

  Signals (4h bars, daily regime):
    * universe: USDT perps, crypto only, >=60d history, 30d median $vol >= $5M
    * daily regime (1-day info lag): close > EMA50 AND mean risk-adjusted
      momentum tanh(1.5 * r_L / (sigma20 * sqrt(L))), L in {20, 60}, > 0.10
    * entry signal: 4h close breaks above the prior 42-bar (7-day) high
    * NO volatility-compression or volume filter (both tested, both hurt)
    * long-only: every short variant tested had negative expectancy

  Execution:
    * limit order at the broken level, working 18 bars (3 days), maker fill
    * initial stop 3.5 ATR(14, 4h) below fill; chandelier trail 5 ATR below
      highest close since entry; both observed at 4h close, exited next open
    * intrabar disaster stop 3 ATR beyond the close-stop
    * no time stop; 6-bar re-entry cooldown per symbol

  Portfolio:
    * risk 0.375% of equity per trade (1R), max 20 positions, heat cap 7.5%,
      gross cap 2x, per-position notional cap 30%
    * costs: maker 2bp entries, taker+slip 7bp exits, extra 5bp stop slip,
      funding 1bp/8h paid on long notional
    * optional overlay (off by default): drawdown throttle dd_k=2 — halves
      new-trade risk at a 25% drawdown; cuts maxDD ~7pts for ~0.17 Sharpe

Usage:
  python3 run_swing.py          # IS only (default, safe to iterate)
  python3 run_swing.py --oos    # + the one-shot OOS window (2025-01 ->)
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import swing_signals
from swing_engine import Config, run, metrics, fmt, trade_stats, monthly

IS_END = pd.Timestamp("2024-12-31", tz="UTC")
START = pd.Timestamp("2022-08-01", tz="UTC")     # 4h warm-up after 2022-05-25 data start

FINAL_SIG = dict(N=42, ts_thr=0.10, require_compression=False)
FINAL_CFG = dict(risk_frac=0.00375, stop_atr=3.5, trail_atr=5.0, tmax_bars=100000,
                 max_positions=20, heat_cap=0.075, gross_cap=2.0,
                 allow_short=False, entry_mode="retest", stop_mode="close",
                 retest_bars=18, disaster_atr=3.0)


def windows(base):
    idx = base.index
    i0 = int(np.searchsorted(idx, START))
    i_split = int(np.searchsorted(idx, IS_END, side="right"))
    return i0, i_split, len(idx)


def report(net, trades, gross, label):
    print(fmt(metrics(net, label)))
    ts = trade_stats(trades)
    if ts:
        print(f"{'':>34}  trades={ts['n_trades']} win={ts['win_rate']*100:.0f}% "
              f"avgR={ts['avg_R']:+.2f} (w{ts['avg_win_R']:+.2f}/l{ts['avg_loss_R']:+.2f}) "
              f"medBars={ts['med_bars']:.0f} avgGross={gross.mean():.2f}x")
    m = monthly(net)
    if len(m):
        print(f"{'':>34}  monthly mean={m.mean()*100:+.1f}% med={m.median()*100:+.1f}% "
              f"+{(m>0).mean()*100:.0f}% best={m.max()*100:+.1f}% worst={m.min()*100:+.1f}%")
    for y in sorted(net.index.year.unique()):
        yr = net[net.index.year == y]
        if len(yr) > 30:
            print(f"{'':>34}  {fmt(metrics(yr, str(y)))}")
    print()


if __name__ == "__main__":
    oos = "--oos" in sys.argv
    print("loading data / building signals ...")
    base = swing_signals.Base()
    sig = base.build(**FINAL_SIG)
    i0, i_split, iT = windows(base)
    print(f"universe={len(base.symbols)}  4h bars: "
          f"IS=[{base.index[i0].date()}..{base.index[i_split-1].date()}] "
          f"OOS=[{base.index[i_split].date()}..{base.index[iT-1].date()}]\n")

    cfg = Config(**FINAL_CFG)
    out_is = run(sig, cfg, start=i0, end=i_split)
    report(out_is["net"], out_is["trades"], out_is["gross"], "CBS v2 [IS]")

    if oos:
        out_oos = run(sig, cfg, start=i_split, end=iT)
        report(out_oos["net"], out_oos["trades"], out_oos["gross"], "CBS v2 [OOS]")
        out_full = run(sig, cfg, start=i0, end=iT)
        report(out_full["net"], out_full["trades"], out_full["gross"], "CBS v2 [FULL]")

        os.makedirs("results", exist_ok=True)
        out_full["equity"].to_csv("results/swing_equity_full.csv")
        out_full["trades"].to_csv("results/swing_trades_full.csv", index=False)
        print("wrote results/swing_equity_full.csv, results/swing_trades_full.csv")
