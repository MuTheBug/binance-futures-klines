"""Offline replay: run the REAL bot pipeline (strategy on a rolling window -> order planning ->
paper fills with fees/slippage, min-notional and rebalance band) day by day over the repo CSVs,
and compare it with the idealized vectorized backtest of the same weights.

This is the end-to-end check that the production code trades what the research tested.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from .brokers import PaperBroker
from .config import Config
from .data import CsvData
from .execution import plan_orders
from .strategy import FUNDING_ANNUAL, compute_targets, compute_weight_history, simulate_net

log = logging.getLogger(__name__)


def _stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    return {"sharpe": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0,
            "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else 0.0,
            "max_dd": float((eq / eq.cummax() - 1).min()), "total": float(eq.iloc[-1] - 1)}


def replay(cfg: Config, start: str, end: str | None = None, data: CsvData | None = None,
           funding_annual: float = FUNDING_ANNUAL, progress=True) -> dict:
    data = data or CsvData(cfg.paper.csv_dir, cfg.strategy.exclude)
    filters = data.filters()
    syms = data.universe()
    broker = PaperBroker.fresh(cfg.paper.initial_equity, cfg.paper.fee, cfg.paper.slippage)
    days = data.all.close.loc[start:end].index
    eq_rows, n_orders, prev = [], 0, None
    for i, d in enumerate(days):
        now = d + pd.Timedelta(days=1, minutes=5)
        bars = data.daily_bars(syms, cfg.strategy.history_bars, now)
        prices = data.mark_prices(d)
        if prev is not None:      # generic funding over the day just held (longs pay, matches research)
            net_notional = sum(q * prices.get(s, 0.0) for s, q in broker.positions().items())
            broker.book["wallet"] -= net_notional * funding_annual / 365
            broker.simulate_stops(bars, prev, cfg.risk.stop_loss_pct)
        out = compute_targets(bars.close, bars.volume, cfg.strategy)
        equity = broker.equity(prices)
        plan = plan_orders(out.weights, equity, broker.positions(), prices, filters, cfg.risk)
        for o in plan.orders:
            broker.execute(o)
        n_orders += len(plan.orders)
        eq_rows.append((d, equity))
        prev = d
        if progress and i % 50 == 0:
            log.info("replay %s equity %.2f positions %d", d.date(), equity, len(broker.positions()))
    eq = pd.Series(dict(eq_rows)).sort_index()
    bot_ret = eq.pct_change().dropna()

    # idealized: same weights, vectorized, continuous rebalancing, no min-notional/band/caps
    full_close, full_vol = data.all.close.loc[:end], data.all.volume.loc[:end]
    blend = compute_weight_history(full_close, full_vol, cfg.strategy)[0]
    ret = full_close[blend.columns].pct_change(fill_method=None)
    ideal = simulate_net(blend * cfg.risk.leverage, ret).loc[bot_ret.index]
    return {"equity": eq, "bot": _stats(bot_ret), "ideal": _stats(ideal),
            "corr": float(bot_ret.corr(ideal)), "orders": n_orders, "days": len(days),
            "fees": broker.book["fees"], "final_positions": broker.positions()}
