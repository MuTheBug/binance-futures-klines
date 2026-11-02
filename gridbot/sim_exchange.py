"""In-process simulator of the Binance USDT-M futures API subset the grid bot uses.

One-way position mode, cross margin. Paper trading feeds it live public prices; offline replay
feeds it historical bars; tests feed it scripted paths. It exposes the SAME method names as
bot.binance.BinanceFuturesClient, so the grid engine cannot tell paper from live.

Semantics modelled:
  * LIMIT orders with timeInForce=GTX (post-only): rejected with code -5022 if they would fill
    immediately; otherwise rest and fill at their limit price when the price trades to them
    (optionally `through` beyond, to model queue position). Maker fee.
  * MARKET orders: fill at the current price +/- slippage. Taker fee. reduceOnly is clamped.
  * Filters: step size, min qty, min notional (not for reduce-only), tick size.
  * Margin: rejects (-2019) orders that would push |position| + resting same-direction orders
    above equity * leverage. Liquidation when equity < maintenance margin.
  * Funding: apply_funding(symbol, rate) charges longs / pays shorts at the current price.
"""
from __future__ import annotations
import itertools
from decimal import Decimal

from bot.binance import BinanceAPIError, SymbolFilters


class SimExchange:
    def __init__(self, filters: dict[str, SymbolFilters], balance: float, leverage: int = 5,
                 maker_fee: float = 0.0002, taker_fee: float = 0.0005, slippage: float = 0.0005,
                 through: float = 0.0, maint_rate: float = 0.01, clock=None):
        self.filters = filters
        self.wallet = float(balance)
        self.leverage = {s: leverage for s in filters}
        self.maker_fee, self.taker_fee, self.slippage = maker_fee, taker_fee, slippage
        self.through, self.maint_rate = through, maint_rate
        self.clock = clock or (lambda: 0)
        self.price: dict[str, float] = {}
        self.pos: dict[str, float] = {s: 0.0 for s in filters}
        self.entry: dict[str, float] = {s: 0.0 for s in filters}
        self.orders: dict[str, dict] = {}               # clientOrderId -> open order
        self.trades: list[dict] = []
        self.fees = 0.0
        self.funding = 0.0
        self.realized = 0.0
        self.liquidated = False
        self._ids = itertools.count(1)

    # ------------------------------------------------------------------ helpers
    def _f(self, symbol) -> SymbolFilters:
        if symbol not in self.filters:
            raise BinanceAPIError(400, -1121, "Invalid symbol.")
        return self.filters[symbol]

    def equity(self) -> float:
        return self.wallet + sum(q * (self.price.get(s, self.entry[s]) - self.entry[s])
                                 for s, q in self.pos.items() if q)

    def _apply_fill(self, symbol, side, qty, price, fee_rate, order_id, client_id, maker):
        dq = qty if side == "BUY" else -qty
        q0, e0 = self.pos[symbol], self.entry[symbol]
        realized = 0.0
        if q0 == 0 or (q0 > 0) == (dq > 0):
            q1 = q0 + dq
            self.entry[symbol] = (q0 * e0 + dq * price) / q1
        else:
            closing = min(abs(dq), abs(q0))
            realized = closing * (price - e0) * (1 if q0 > 0 else -1)
            q1 = q0 + dq
            if abs(q1) < 1e-12:
                q1 = 0.0
                self.entry[symbol] = 0.0
            elif (q1 > 0) != (q0 > 0):
                self.entry[symbol] = price
        self.pos[symbol] = q1
        fee = qty * price * fee_rate
        self.wallet += realized - fee
        self.realized += realized
        self.fees += fee
        self.trades.append({"symbol": symbol, "side": side, "qty": qty, "price": price, "fee": fee,
                            "realizedPnl": realized, "orderId": order_id, "clientOrderId": client_id,
                            "maker": maker, "time": self.clock(), "id": len(self.trades) + 1})

    def _open_increase_notional(self, symbol, extra_side=None, extra_qty=0.0, extra_price=0.0):
        """|position| plus resting orders that would increase it, worst side, in quote."""
        q = self.pos[symbol]
        px = self.price.get(symbol, 0.0)
        buys = sum(float(o["origQty"]) * float(o["price"]) for o in self.orders.values()
                   if o["symbol"] == symbol and o["side"] == "BUY" and not o["reduceOnly"])
        sells = sum(float(o["origQty"]) * float(o["price"]) for o in self.orders.values()
                    if o["symbol"] == symbol and o["side"] == "SELL" and not o["reduceOnly"])
        if extra_side == "BUY":
            buys += extra_qty * extra_price
        elif extra_side == "SELL":
            sells += extra_qty * extra_price
        long_side = max(q, 0) * px + max(0.0, buys - max(-q, 0) * px)
        short_side = max(-q, 0) * px + max(0.0, sells - max(q, 0) * px)
        return max(long_side, short_side)

    def _margin_ok(self, symbol, side, qty, price) -> bool:
        need = 0.0
        for s in self.filters:
            if s == symbol:
                need += self._open_increase_notional(s, side, qty, price) / self.leverage[s]
            else:
                need += self._open_increase_notional(s) / self.leverage[s]
        return need <= self.equity() + 1e-9

    def _check_liquidation(self):
        maint = sum(abs(q) * self.price.get(s, 0.0) for s, q in self.pos.items()) * self.maint_rate
        if any(self.pos.values()) and self.equity() <= maint:
            self.liquidated = True
            for o in list(self.orders):
                self.orders.pop(o)
            for s, q in list(self.pos.items()):
                if q:
                    self._apply_fill(s, "SELL" if q > 0 else "BUY", abs(q), self.price[s],
                                     self.taker_fee, -1, "liquidation", False)

    # ------------------------------------------------------------------ price feed
    def set_price(self, symbol: str, price: float):
        """Move the price (a continuous path segment) and fill resting orders it crosses."""
        p0 = self.price.get(symbol)
        self.price[symbol] = price
        if p0 is None or price == p0:
            return
        th = self.through
        if price < p0:
            hits = sorted((o for o in self.orders.values() if o["symbol"] == symbol and o["side"] == "BUY"
                           and float(o["price"]) * (1 - th) >= price - 1e-12 * float(o["price"])),
                          key=lambda o: -float(o["price"]))
        else:
            hits = sorted((o for o in self.orders.values() if o["symbol"] == symbol and o["side"] == "SELL"
                           and float(o["price"]) / (1 - th) <= price + 1e-12 * float(o["price"])),
                          key=lambda o: float(o["price"]))
        for o in hits:
            if o["clientOrderId"] not in self.orders:
                continue
            qty = float(o["origQty"])
            if o["reduceOnly"]:
                cur = self.pos[symbol]
                if cur == 0 or (cur > 0) == (o["side"] == "BUY"):
                    self.orders.pop(o["clientOrderId"])
                    continue
                qty = min(qty, abs(cur))
            self.orders.pop(o["clientOrderId"])
            self._apply_fill(symbol, o["side"], qty, float(o["price"]), self.maker_fee, o["orderId"],
                             o["clientOrderId"], True)
        self._check_liquidation()

    def apply_funding(self, symbol: str, rate: float):
        q = self.pos.get(symbol, 0.0)
        if q:
            paid = q * self.price[symbol] * rate
            self.wallet -= paid
            self.funding += paid

    # ------------------------------------------------------------------ API subset
    def sync_time(self):
        return 0

    def dual_side_position(self) -> bool:
        return False

    def set_leverage(self, symbol, leverage):
        self._f(symbol)
        self.leverage[symbol] = int(leverage)
        return {"symbol": symbol, "leverage": int(leverage)}

    def set_margin_type(self, symbol, margin_type):
        return None

    def premium_index(self):
        return [{"symbol": s, "markPrice": str(p), "lastFundingRate": "0.0001"} for s, p in self.price.items()]

    def account(self):
        eq = self.equity()
        used = sum(self._open_increase_notional(s) / self.leverage[s] for s in self.filters)
        return {"totalMarginBalance": str(eq), "totalWalletBalance": str(self.wallet),
                "availableBalance": str(max(0.0, eq - used))}

    def position_risk(self):
        return [{"symbol": s, "positionAmt": str(q), "entryPrice": str(self.entry[s]),
                 "markPrice": str(self.price.get(s, 0.0)),
                 "unRealizedProfit": str(q * (self.price.get(s, 0.0) - self.entry[s]))}
                for s, q in self.pos.items() if q]

    def open_orders(self, symbol=None):
        return [dict(o) for o in self.orders.values() if symbol is None or o["symbol"] == symbol]

    def cancel_order(self, symbol, client_order_id):
        o = self.orders.get(client_order_id)
        if o is None or o["symbol"] != symbol:
            raise BinanceAPIError(400, -2011, "Unknown order sent.")
        self.orders.pop(client_order_id)
        return {**o, "status": "CANCELED"}

    def cancel_all_open_orders(self, symbol):
        for k in [k for k, o in self.orders.items() if o["symbol"] == symbol]:
            self.orders.pop(k)
        return {"code": 200, "msg": "The operation of cancel all open order is done."}

    def user_trades(self, symbol, from_id=None):
        return [t for t in self.trades if t["symbol"] == symbol and (from_id is None or t["id"] >= from_id)]

    def new_order(self, **p):
        if self.liquidated:
            raise BinanceAPIError(400, -2019, "Margin is insufficient.")
        symbol, side, typ = p["symbol"], p["side"], p["type"]
        f = self._f(symbol)
        qty = Decimal(str(p["quantity"]))
        if qty <= 0 or (qty % f.step) != 0 or qty < f.min_qty or qty > f.max_qty:
            raise BinanceAPIError(400, -1111, "Precision is over the maximum defined for this asset.")
        cid = p.get("newClientOrderId") or f"sim{next(self._ids)}"
        if cid in self.orders:
            raise BinanceAPIError(400, -4116, "ClientOrderId is duplicated.")
        reduce_only = str(p.get("reduceOnly", "false")).lower() == "true"
        px_now = self.price.get(symbol)
        if px_now is None:
            raise BinanceAPIError(400, -1121, "No price yet.")
        q = float(qty)
        oid = next(self._ids)
        if typ == "MARKET":
            if reduce_only:
                cur = self.pos[symbol]
                if cur == 0 or (cur > 0) == (side == "BUY"):
                    raise BinanceAPIError(400, -2022, "ReduceOnly Order is rejected.")
                q = min(q, abs(cur))
            elif q * px_now < f.min_notional:
                raise BinanceAPIError(400, -4164, f"Order's notional must be no smaller than {f.min_notional}")
            fill_px = px_now * (1 + self.slippage if side == "BUY" else 1 - self.slippage)
            if not reduce_only and not self._margin_ok(symbol, side, q, fill_px):
                raise BinanceAPIError(400, -2019, "Margin is insufficient.")
            self._apply_fill(symbol, side, q, fill_px, self.taker_fee, oid, cid, False)
            self._check_liquidation()
            return {"orderId": oid, "clientOrderId": cid, "symbol": symbol, "status": "FILLED",
                    "executedQty": str(q), "avgPrice": str(fill_px), "side": side, "type": "MARKET"}
        if typ != "LIMIT":
            raise BinanceAPIError(400, -1116, "Invalid orderType.")
        price = Decimal(str(p["price"]))
        if (price % f.tick) != 0:
            raise BinanceAPIError(400, -4014, "Price not increased by tick size.")
        pf = float(price)
        if not reduce_only and q * pf < f.min_notional:
            raise BinanceAPIError(400, -4164, f"Order's notional must be no smaller than {f.min_notional}")
        crosses = (side == "BUY" and pf >= px_now) or (side == "SELL" and pf <= px_now)
        if p.get("timeInForce") == "GTX" and crosses:
            raise BinanceAPIError(400, -5022, "Due to the order could not be executed as maker, the Post Only order will be rejected.")
        if crosses:                                         # plain GTC that crosses: fills as taker
            self._apply_fill(symbol, side, q, px_now, self.taker_fee, oid, cid, False)
            return {"orderId": oid, "clientOrderId": cid, "symbol": symbol, "status": "FILLED",
                    "executedQty": str(q), "avgPrice": str(px_now), "side": side, "type": "LIMIT"}
        if not reduce_only and not self._margin_ok(symbol, side, q, pf):
            raise BinanceAPIError(400, -2019, "Margin is insufficient.")
        o = {"orderId": oid, "clientOrderId": cid, "symbol": symbol, "side": side, "type": "LIMIT",
             "price": str(price), "origQty": str(qty), "executedQty": "0", "status": "NEW",
             "reduceOnly": reduce_only, "timeInForce": p.get("timeInForce", "GTC")}
        self.orders[cid] = o
        return dict(o)

    # ------------------------------------------------------------------ persistence (paper mode)
    def to_dict(self) -> dict:
        return {"wallet": self.wallet, "pos": self.pos, "entry": self.entry, "orders": self.orders,
                "price": self.price, "fees": self.fees, "funding": self.funding, "realized": self.realized,
                "liquidated": self.liquidated, "leverage": self.leverage, "trades": self.trades[-500:],
                "next_id": next(self._ids)}

    def load_dict(self, d: dict):
        self.wallet = float(d["wallet"])
        self.pos.update({k: float(v) for k, v in d["pos"].items() if k in self.pos})
        self.entry.update({k: float(v) for k, v in d["entry"].items() if k in self.entry})
        self.orders = {k: v for k, v in d["orders"].items() if v["symbol"] in self.filters}
        self.price.update({k: float(v) for k, v in d["price"].items()})
        self.fees, self.funding, self.realized = d["fees"], d["funding"], d["realized"]
        self.liquidated = d["liquidated"]
        self.leverage.update(d.get("leverage", {}))
        self.trades = d.get("trades", [])
        self._ids = itertools.count(int(d.get("next_id", 1)))
