"""Offline replay: run the REAL bot pipeline (strategy on a rolling window -> order planning ->
paper fills with fees/slippage, min-notional and rebalance band) day by day over the repo CSVs,
and compare it with the idealized vectorized backtest of the same weights.

This is the end-to-end check that the production code trades what the research tested.
"""
from __future__ import annotations
import logging
from decimal import Decimal

import numpy as np
import pandas as pd

from .brokers import PaperBroker
from .config import Config
from .data import CsvData
from .execution import Order, plan_orders
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
           funding_annual: float = FUNDING_ANNUAL, progress=True, funding: pd.DataFrame | None = None,
           universe_fn=None) -> dict:
    """funding: optional real funding per symbol and HOLDING day (rate summed over the day's events;
    longs pay when positive) - replaces the generic drag. universe_fn(day) -> symbols tradable that
    day (what the exchange would list); None = every symbol in the data."""
    data = data or CsvData(cfg.paper.csv_dir, cfg.strategy.exclude)
    filters = data.filters()
    all_syms = data.universe()
    broker = PaperBroker.fresh(cfg.paper.initial_equity, cfg.paper.fee, cfg.paper.slippage)
    days = data.all.close.loc[start:end].index
    eq_rows, n_orders, prev = [], 0, None
    held_gross, target_gross, n_pos, n_settled, funding_paid = [], [], [], 0, 0.0
    for i, d in enumerate(days):
        now = d + pd.Timedelta(days=1, minutes=5)
        syms = all_syms if universe_fn is None else list(universe_fn(d))
        bars = data.daily_bars(syms, cfg.strategy.history_bars, now)
        prices = data.mark_prices(d)
        for s, q in list(broker.positions().items()):          # contract stopped trading: settle at
            if s not in prices:                                # its last price (delisting)
                last = data.all.close[s].loc[:d].dropna()
                broker.execute(Order(s, "SELL" if q > 0 else "BUY", Decimal(str(abs(q))), True, "delisted",
                                     float(last.iloc[-1])))
                n_settled += 1
        if prev is not None:      # funding over the day just held (longs pay when the rate is positive)
            pos = broker.positions()
            if funding is None:   # generic drag, matches the original research
                paid = sum(q * prices.get(s, 0.0) for s, q in pos.items()) * funding_annual / 365
            else:
                row = funding.loc[d] if d in funding.index else {}
                paid = sum(q * prices.get(s, 0.0) * float(np.nan_to_num(row.get(s, 0.0))) for s, q in pos.items())
            broker.book["wallet"] -= paid
            funding_paid += paid
            broker.simulate_stops(bars, prev, cfg.risk.stop_loss_pct)
        out = compute_targets(bars.close, bars.volume, cfg.strategy)
        equity = broker.equity(prices)
        plan = plan_orders(out.weights, equity, broker.positions(), prices, filters, cfg.risk)
        for o in plan.orders:
            broker.execute(o)
        n_orders += len(plan.orders)
        pos = broker.positions()
        held_gross.append(sum(abs(q) * prices.get(s, 0.0) for s, q in pos.items()) / max(equity, 1e-9))
        target_gross.append(float(out.weights.abs().sum()) * cfg.risk.leverage)
        n_pos.append(len(pos))
        eq_rows.append((d, equity))
        prev = d
        if progress and i % 50 == 0:
            log.info("replay %s equity %.2f positions %d", d.date(), equity, len(broker.positions()))
    eq = pd.Series(dict(eq_rows)).sort_index()
    bot_ret = eq.pct_change().dropna()
    res = {"equity": eq, "bot": _stats(bot_ret), "orders": n_orders, "days": len(days),
           "fees": broker.book["fees"], "funding_paid": funding_paid, "settled": n_settled,
           "final_positions": broker.positions(),
           "avg_positions": float(np.mean(n_pos)) if n_pos else 0.0,
           "held_vs_target_gross": float(np.sum(held_gross) / max(np.sum(target_gross), 1e-9))}
    if universe_fn is None and funding is None:
        # idealized: same weights, vectorized, continuous rebalancing, no min-notional/band/caps
        full_close, full_vol = data.all.close.loc[:end], data.all.volume.loc[:end]
        blend = compute_weight_history(full_close, full_vol, cfg.strategy)[0]
        ret = full_close[blend.columns].pct_change(fill_method=None)
        ideal = simulate_net(blend * cfg.risk.leverage, ret).loc[bot_ret.index]
        res.update(ideal=_stats(ideal), corr=float(bot_ret.corr(ideal)))
    return res
