"""Turn target weights into exchange-valid orders.

Rules:
  * notional = weight * equity * leverage, then per-symbol cap, then gross cap (scale down).
  * Quantities are rounded TOWARD ZERO to the symbol step (never overshoot risk).
  * Targets below the exchange min-notional become 0 (can't hold a position that small).
  * Changes smaller than rebalance_band * equity are skipped, but full closes always go through.
  * A sign flip is split into a reduce-only close plus a fresh open.
  * Reduce-only orders are returned first so they free margin for the opens.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN

import pandas as pd

from .binance import SymbolFilters
from .config import RiskCfg

log = logging.getLogger(__name__)


@dataclass
class Order:
    symbol: str
    side: str                 # BUY / SELL
    qty: Decimal
    reduce_only: bool
    reason: str
    price_hint: float = 0.0

    @property
    def signed_qty(self) -> float:
        return float(self.qty) if self.side == "BUY" else -float(self.qty)


@dataclass
class Plan:
    orders: list[Order] = field(default_factory=list)
    target_qty: dict[str, float] = field(default_factory=dict)
    target_notional: dict[str, float] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    gross_scale: float = 1.0


def round_qty(q: float, step: Decimal) -> Decimal:
    """Round |q| down to the step, keep the sign."""
    d = (Decimal(str(abs(q))) / step).to_integral_value(rounding=ROUND_DOWN) * step
    return d if q >= 0 else -d


def target_notionals(weights: pd.Series, equity: float, risk: RiskCfg) -> tuple[pd.Series, float]:
    n = weights.astype(float) * equity * risk.leverage
    cap = risk.max_position * equity
    n = n.clip(lower=-cap, upper=cap)
    gross = n.abs().sum()
    scale = 1.0
    if gross > risk.max_gross * equity > 0:
        scale = risk.max_gross * equity / gross
        n = n * scale
    return n, scale


def plan_orders(weights: pd.Series, equity: float, positions: dict[str, float],
                prices: dict[str, float], filters: dict[str, SymbolFilters], risk: RiskCfg) -> Plan:
    plan = Plan()
    if equity <= 0:
        raise ValueError(f"non-positive equity {equity}")
    notion, plan.gross_scale = target_notionals(weights, equity, risk)
    band = risk.rebalance_band * equity
    reduces, opens = [], []
    for sym in sorted(set(notion.index) | {s for s, q in positions.items() if q}):
        cur = float(positions.get(sym, 0.0))
        f, px = filters.get(sym), prices.get(sym)
        if f is None or not px or px <= 0:
            if cur or abs(notion.get(sym, 0.0)) > 0:
                plan.skipped.append(f"{sym}: no filters/price (not tradable?)")
            continue
        tn = float(notion.get(sym, 0.0))
        if abs(tn) < f.min_notional:
            tn = 0.0
        tq = float(round_qty(tn / px, f.step))
        if abs(tq) < float(f.min_qty):
            tq = 0.0
        plan.target_qty[sym], plan.target_notional[sym] = tq, tq * px
        dq = tq - cur
        if abs(dq) < 1e-15:
            continue
        if tq == 0.0:                                    # full close: always execute
            reduces.append(_order(sym, -cur, True, "close", f, px))
            continue
        if abs(dq) * px < band:
            plan.skipped.append(f"{sym}: change {dq * px:+.2f} below band")
            continue
        if cur != 0 and (cur > 0) != (tq > 0):           # flip: close then open
            reduces.append(_order(sym, -cur, True, "flip-close", f, px))
            opens.append(_order(sym, tq, False, "flip-open", f, px))
        elif cur != 0 and abs(tq) < abs(cur):            # same side, smaller
            reduces.append(_order(sym, dq, True, "reduce", f, px))
        else:                                            # open or increase
            if abs(dq) * px < f.min_notional:
                plan.skipped.append(f"{sym}: increase {dq * px:.2f} below min notional")
                continue
            opens.append(_order(sym, dq, False, "open" if cur == 0 else "increase", f, px))
    plan.orders = [o for o in reduces + opens if o is not None]
    return plan


def _order(sym, signed_qty, reduce_only, reason, f: SymbolFilters, px) -> Order | None:
    q = round_qty(abs(signed_qty), f.step)
    if q <= 0:
        return None
    if q > f.max_qty:
        log.warning("%s qty %s above max %s; clipping", sym, q, f.max_qty)
        q = f.max_qty
    return Order(sym, "BUY" if signed_qty > 0 else "SELL", q, reduce_only, reason, px)
