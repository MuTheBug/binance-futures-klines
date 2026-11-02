"""`python -m gridbot replay`: run the real engine + SimExchange over the repo's hourly CSVs and
compare with the research simulator on the same bars and parameters."""
from __future__ import annotations
import os

import numpy as np
import pandas as pd

from .engine import GridPlan
from .replay import fine_filters, replay
from .strategy import LOOKBACKS, hourly_sigma, trend_strength

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_csv(csv_dir: str, symbol: str) -> pd.DataFrame:
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    path = os.path.join(csv_dir, f"{base}_USDT_1h.csv")
    df = pd.read_csv(path).drop_duplicates("timestamp")
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["open", "high", "low", "close"]].astype(float).sort_index()


def make_trend_fn(bars: pd.DataFrame):
    """Daily trend strength known at hour t: uses the daily close of day D from D 23:00 on."""
    daily = bars["close"].resample("D").last()

    def trend(t):
        d = daily.loc[: t.floor("D") - pd.Timedelta(days=0 if t.hour == 23 else 1)].to_numpy()
        return trend_strength(list(d))
    return trend


def make_regime_fn(cfg, bars: pd.DataFrame):
    """The daily trend filter the live runner applies: False flattens a running grid."""
    trend = make_trend_fn(bars)

    def regime(symbol, t):
        tr = trend(t)
        return bool(np.isfinite(tr) and abs(tr) < cfg.grid.trend_threshold)
    return regime


def make_plan_fn(cfg, bars: pd.DataFrame):
    """Spacing/bias from data known at time t (no look-ahead), same formulas as live."""
    closes = bars["close"]
    trend_at = make_trend_fn(bars)
    gc = cfg.grid

    def plan(symbol, t):
        hist = closes.loc[:t].to_numpy()
        if gc.spacing == "fixed":
            g = gc.spacing_fixed
        else:
            s = hourly_sigma(list(hist))
            if not np.isfinite(s):
                return GridPlan(False, reason="warm-up")
            g = min(max(gc.spacing_k * s, gc.spacing_min), gc.spacing_max)
        bias = gc.bias
        if gc.bias == "trend" or gc.regime_filter:
            tr = trend_at(t)
            if not np.isfinite(tr):
                return GridPlan(False, reason="warm-up")
            if gc.regime_filter and abs(tr) >= gc.trend_threshold:
                return GridPlan(False, reason="trending")
            if gc.bias == "trend":
                bias = "long" if tr > gc.trend_threshold else ("short" if tr < -gc.trend_threshold else "neutral")
        return GridPlan(True, g=g, bias=bias)
    return plan


def replay_cmd(cfg, args):
    csv_dir = ROOT if cfg.paper.csv_dir == "." else cfg.paper.csv_dir
    df = load_csv(csv_dir, args.symbol)
    warm = pd.Timestamp(args.start, tz="UTC") - pd.Timedelta(days=max(LOOKBACKS) + 40)
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(hours=23) if args.end else None
    df = df.loc[warm:end]
    plan = make_plan_fn(cfg, df)
    start = pd.Timestamp(args.start, tz="UTC")

    def plan_live(s, t):
        return plan(s, t) if t >= start else GridPlan(False, reason="before start")
    eq, ex = replay({args.symbol: df}, plan_live, balance=args.balance, n=cfg.grid.levels,
                    exposure=cfg.risk.exposure, exit_rule=cfg.grid.exit_rule, exit_buffer=cfg.grid.exit_buffer,
                    maker_fee=cfg.paper.maker_fee, taker_fee=cfg.paper.taker_fee, slippage=cfg.paper.slippage,
                    through=cfg.paper.through, filters=fine_filters([args.symbol], min_notional=5.0),
                    leverage=cfg.risk.exchange_leverage, cooldown_s=cfg.grid.cooldown_minutes * 60,
                    recenter_hours=cfg.grid.recenter_hours,
                    regime_fn=make_regime_fn(cfg, df) if cfg.grid.regime_filter else None)
    eq = eq.loc[start:]
    d = (eq.resample("D").last().pct_change()).dropna()
    yrs = max(len(d) / 365, 1e-9)
    sharpe = d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else float("nan")
    mdd = (eq / eq.cummax() - 1).min()
    bh = df["close"].loc[start:]
    print(f"replay {args.symbol} {eq.index[0].date()}..{eq.index[-1].date()} start {args.balance:.2f} USDT")
    print(f"  grid bot : final {eq.iloc[-1]:,.2f}  return {eq.iloc[-1] / args.balance - 1:+.1%}  "
          f"CAGR {(eq.iloc[-1] / args.balance) ** (1 / yrs) - 1:+.1%}  Sharpe {sharpe:.2f}  maxDD {mdd:.1%}")
    print(f"  fills {len(ex.trades)}  fees {ex.fees:.2f}  funding {ex.funding:+.2f}  liquidated {ex.liquidated}")
    print(f"  buy & hold {args.symbol}: {bh.iloc[-1] / bh.iloc[0] - 1:+.1%}")
    return 0
