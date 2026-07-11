"""
Portfolio reconciliation + execution.

Turns target weights (fraction of equity per symbol) into the minimal set of futures
orders that moves the current book to the target, with safety controls:
  * dust filter      — skip orders below `min_order_usdt`
  * no-churn band    — skip a name if it has barely drifted from target
  * reduce-only logic — closes/reductions use reduceOnly; side-flips do not (so they fill)
  * lot-size rounding — quantities rounded down to the symbol's stepSize
Everything respects the client's DRY_RUN flag (orders are reported, not sent).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Order:
    symbol: str
    side: str          # BUY / SELL
    qty: float
    reduce_only: bool
    notional: float
    reason: str


def reconcile(targets, equity, positions, prices, client, cfg):
    """Return a list[Order] to move `positions` toward `targets`.
    targets   : {symbol: weight}        positions: {symbol: signed_amount}
    prices    : {symbol: price}         equity: float (USDT)
    """
    orders = []
    symbols = set(targets) | set(positions)
    for sym in sorted(symbols):
        price = prices.get(sym)
        if not price or price <= 0:
            continue
        w = targets.get(sym, 0.0)
        cur = positions.get(sym, 0.0)
        tgt_notional = w * equity
        tgt_qty = tgt_notional / price
        delta_qty = tgt_qty - cur
        delta_notional = delta_qty * price
        if abs(delta_qty) <= 0:
            continue

        is_close = (w == 0.0 and cur != 0.0)
        is_new = (cur == 0.0)
        # effective dust floor = max(config min, the symbol's exchange minNotional)
        sym_min = client._filters.get(sym, {}).get("minNotional", 0.0) if hasattr(client, "_filters") else 0.0
        floor = max(cfg.min_order_usdt, sym_min)
        # no-churn band: hold if the position barely moved (but always act on closes/new)
        if not is_close and not is_new:
            if abs(delta_notional) < max(floor, cfg.rebalance_band * abs(tgt_notional)):
                continue
        elif abs(delta_notional) < floor and not is_close:
            continue

        qty = client.round_qty(sym, abs(delta_qty))
        if qty <= 0:
            # still try to fully close a residual position even if tiny
            if is_close:
                qty = client.round_qty(sym, abs(cur))
                if qty <= 0:
                    continue
            else:
                continue

        side = "BUY" if delta_qty > 0 else "SELL"
        same_sign_shrink = (cur != 0 and (cur > 0) == (tgt_qty > 0) and abs(tgt_qty) < abs(cur))
        reduce_only = bool(is_close or same_sign_shrink)
        reason = "open" if is_new else ("close" if is_close else ("trim" if reduce_only else "adjust"))
        orders.append(Order(sym, side, qty, reduce_only, qty * price, reason))
    return orders


def _execute_maker(o, client, cfg, log=print):
    """Try a post-only limit at the passive touch; on timeout/rejection fall
    back to a market order for the remainder. Saves ~3-5bps/side of
    taker fee + spread on a book whose signals move over days, not seconds."""
    import time as _t
    bid, ask = client.book_ticker(o.symbol)
    px = client.round_price(o.symbol, bid if o.side == "BUY" else ask)
    res = client.place_post_only(o.symbol, o.side, o.qty, px,
                                 reduce_only=o.reduce_only)
    oid = res.get("orderId")
    if oid is None:                      # GTX rejected (would cross) or odd reply
        return client.place_order(o.symbol, o.side, o.qty,
                                  reduce_only=o.reduce_only)
    deadline = _t.time() + max(5, getattr(cfg, "maker_wait_s", 45))
    while _t.time() < deadline:
        _t.sleep(3)
        st = client.get_order(o.symbol, oid)
        if st.get("status") == "FILLED":
            return st
        if st.get("status") in ("CANCELED", "EXPIRED", "REJECTED"):
            break
    try:
        client.cancel_order(o.symbol, oid)
    except Exception:
        pass
    st = client.get_order(o.symbol, oid)
    rem = client.round_qty(o.symbol, o.qty - float(st.get("executedQty", 0) or 0))
    if rem > 0:
        return client.place_order(o.symbol, o.side, rem,
                                  reduce_only=o.reduce_only)
    return st


def execute(orders, client, cfg, log=print):
    """Set leverage on touched symbols, then place each order. Returns a result summary."""
    placed, errors = [], []
    touched = sorted({o.symbol for o in orders})
    for sym in touched:
        client.set_leverage(sym, cfg.binance_leverage)
    use_maker = (getattr(cfg, "exec_style", "market") == "maker"
                 and hasattr(client, "book_ticker")
                 and not getattr(client, "dry_run", True))
    for o in orders:
        try:
            if use_maker:
                try:
                    res = _execute_maker(o, client, cfg, log)
                except Exception as me:      # NEVER leave a rebalance unexecuted
                    log(f"  [maker->market] {o.symbol}: {me}")
                    res = client.place_order(o.symbol, o.side, o.qty,
                                             reduce_only=o.reduce_only)
            else:
                res = client.place_order(o.symbol, o.side, o.qty,
                                         reduce_only=o.reduce_only)
            placed.append((o, res))
            tag = "DRY" if res.get("dry_run") else "SENT"
            log(f"  [{tag}] {o.side:4} {o.qty:<12g} {o.symbol:14} ~${o.notional:,.2f} ({o.reason})")
        except Exception as e:
            errors.append((o, str(e)))
            log(f"  [ERR] {o.side} {o.qty} {o.symbol}: {e}")
    return {"placed": placed, "errors": errors, "n": len(placed)}


def flatten(client, cfg, log=print):
    """Close ALL open positions with reduce-only market orders (panic button)."""
    positions = client.positions()
    orders = []
    for sym, amt in positions.items():
        qty = client.round_qty(sym, abs(amt))
        if qty <= 0:
            continue
        side = "SELL" if amt > 0 else "BUY"
        orders.append(Order(sym, side, qty, True, 0.0, "flatten"))
    return execute(orders, client, cfg, log)
