"""Offline replay: drive SimExchange + GridEngine through historical hourly bars.

Each bar is walked as previous close -> O -> L/H -> H/L -> C (same path rule as the research
simulator); the engine steps after every path point (so freshly filled levels get their opposite
order before the price turns), funding is charged hourly, and range exits are checked at closes.
"""
from __future__ import annotations
from decimal import Decimal

import numpy as np
import pandas as pd

from bot.binance import SymbolFilters
from .engine import GridEngine, GridPlan
from .sim_exchange import SimExchange

HOURS_PER_YEAR = 24 * 365


def fine_filters(symbols, min_notional=0.0) -> dict[str, SymbolFilters]:
    d = Decimal("0.00000001")
    return {s: SymbolFilters(s, s, d, d, Decimal("1e15"), d, min_notional) for s in symbols}


def replay(bars: dict[str, pd.DataFrame], plan_fn, *, balance=1.0, n=10, exposure=1.0,
           exit_rule="reset", exit_buffer=1, maker_fee=0.0002, taker_fee=0.0005, slippage=0.0005,
           through=0.0, funding_annual=0.11, filters=None, leverage=100, path="heuristic", cooldown_s=0,
           recenter_hours=0, regime_fn=None):
    """bars: {symbol: DataFrame[open, high, low, close]} on one hourly index (one grid per symbol,
    capital split equally). plan_fn(symbol, t) -> GridPlan. recenter_hours: periodic re-centre at a
    bar close. regime_fn(symbol, t) -> bool: daily trend filter (False flattens at the close).
    Returns equity Series and the exchange."""
    syms = list(bars)
    idx = bars[syms[0]].index
    filters = filters or fine_filters(syms)
    ex = SimExchange(filters, balance, leverage=leverage, maker_fee=maker_fee, taker_fee=taker_fee,
                     slippage=slippage, through=through)
    state = {}
    eng = GridEngine(ex, filters, state, n=n, exposure=exposure, exit_rule=exit_rule, exit_buffer=exit_buffer,
                     cooldown_s=cooldown_s)
    f_h = funding_annual / HOURS_PER_YEAR
    eq = []
    started = {s: False for s in syms}
    for i, t in enumerate(idx):
        ms = int(t.timestamp() * 1000)
        for s in syms:
            o, h, l, c = (bars[s].at[t, k] for k in ("open", "high", "low", "close"))
            if not np.isfinite(c):
                continue
            cap = lambda: ex.equity() / len(syms)        # sized at the moment a grid is (re)created
            if started[s]:
                bull = c >= o
                pts = [o, l, h] if bull == (path == "heuristic") else [o, h, l]
                for j, pt in enumerate(pts):
                    ex.set_price(s, pt)
                    eng.step(s, pt, lambda: plan_fn(s, t), cap, ms, f"{i}{j}", check_exit=False, allow_create=False)
                ex.set_price(s, c)
                if f_h:
                    ex.apply_funding(s, f_h)
                g = eng.grid(s)
                if g and recenter_hours and ms - g.created_ms >= recenter_hours * 3_600_000:
                    ex.cancel_all_open_orders(s)
                    eng._save_grid(None, s)                  # re-created below, around this close
                if regime_fn is not None:
                    state.setdefault("regime_check", {})[s] = bool(regime_fn(s, t))
                eng.step(s, c, lambda: plan_fn(s, t), cap, ms, f"{i}c", check_exit=True)
            else:
                ex.set_price(s, c)
                eng.step(s, c, lambda: plan_fn(s, t), cap, ms, f"{i}c", check_exit=True)
                started[s] = True
        eq.append(ex.equity())
    return pd.Series(eq, index=idx), ex
