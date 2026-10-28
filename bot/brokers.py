"""Brokers: PaperBroker (simulated fills, persisted in state) and LiveBroker (Binance USDT-M).

Both expose:  equity(prices) -> float,  positions() -> {symbol: signed qty},
              execute(order, tag) -> fill dict,  flatten(prices, tag) -> fills
"""
from __future__ import annotations
import logging
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP

from .binance import AmbiguousOrderError, BinanceAPIError, BinanceFuturesClient, SymbolFilters
from .execution import Order

log = logging.getLogger(__name__)


def fmt_dec(d: Decimal) -> str:
    return format(d.normalize(), "f")


# =============================================================================== paper
class PaperBroker:
    """Simulated futures account. `book` is a dict persisted by the runner:
    {"wallet": float, "positions": {sym: {"qty": float, "entry": float}},
     "fees": float, "funding": float, "realized": float}"""

    mode = "paper"

    def __init__(self, book: dict, fee: float, slippage: float):
        self.book = book
        self.book.setdefault("positions", {})
        for k in ("fees", "funding", "realized"):
            self.book.setdefault(k, 0.0)
        self.fee, self.slippage = fee, slippage

    @classmethod
    def fresh(cls, equity: float, fee: float, slippage: float) -> "PaperBroker":
        return cls({"wallet": float(equity), "positions": {}}, fee, slippage)

    def positions(self) -> dict[str, float]:
        return {s: p["qty"] for s, p in self.book["positions"].items() if p["qty"]}

    def entries(self) -> dict[str, float]:
        return {s: p["entry"] for s, p in self.book["positions"].items()}

    def equity(self, prices: dict[str, float]) -> float:
        upnl = 0.0
        for s, p in self.book["positions"].items():
            px = prices.get(s, p["entry"])
            upnl += p["qty"] * (px - p["entry"])
        return self.book["wallet"] + upnl

    def _fill(self, sym: str, dq: float, price: float):
        pos = self.book["positions"].get(sym, {"qty": 0.0, "entry": 0.0})
        q0, e0 = pos["qty"], pos["entry"]
        if q0 == 0 or (q0 > 0) == (dq > 0):
            q1 = q0 + dq
            pos = {"qty": q1, "entry": (q0 * e0 + dq * price) / q1}
        else:
            closing = min(abs(dq), abs(q0))
            pnl = closing * (price - e0) * (1 if q0 > 0 else -1)
            self.book["wallet"] += pnl
            self.book["realized"] += pnl
            q1 = q0 + dq
            if abs(q1) < 1e-12:
                q1 = 0.0
            entry = e0 if (q1 == 0 or (q1 > 0) == (q0 > 0)) else price
            pos = {"qty": q1, "entry": entry}
        fee = abs(dq) * price * self.fee
        self.book["wallet"] -= fee
        self.book["fees"] += fee
        if pos["qty"] == 0:
            self.book["positions"].pop(sym, None)
        else:
            self.book["positions"][sym] = pos
        return fee

    def execute(self, order: Order, tag: str = "") -> dict:
        cur = self.positions().get(order.symbol, 0.0)
        dq = order.signed_qty
        if order.reduce_only:                      # never let reduce-only increase/flip
            if cur == 0 or (cur > 0) == (dq > 0):
                return {"symbol": order.symbol, "status": "REJECTED", "reason": "reduce-only, nothing to reduce"}
            dq = min(abs(dq), abs(cur)) * (1 if dq > 0 else -1)
        px = order.price_hint * (1 + self.slippage if dq > 0 else 1 - self.slippage)
        fee = self._fill(order.symbol, dq, px)
        return {"symbol": order.symbol, "side": order.side, "qty": abs(dq), "price": px, "fee": fee,
                "status": "FILLED", "reduce_only": order.reduce_only, "reason": order.reason}

    def apply_funding(self, rates: dict[str, float], prices: dict[str, float]) -> float:
        """rates: summed funding rate per symbol since last cycle (positive => longs pay)."""
        paid = 0.0
        for s, q in self.positions().items():
            r = rates.get(s, 0.0)
            px = prices.get(s, self.book["positions"][s]["entry"])
            paid += q * px * r
        self.book["wallet"] -= paid
        self.book["funding"] += paid
        return paid

    def simulate_stops(self, bars, since_day, stop_pct: float) -> list[dict]:
        """Catastrophe stops on daily OHLC for bars after since_day (gap-aware fills)."""
        fills = []
        if stop_pct <= 0:
            return fills
        for s, p in list(self.book["positions"].items()):
            if s not in bars.close.columns:
                continue
            q, e = p["qty"], p["entry"]
            stop = e * (1 - stop_pct) if q > 0 else e * (1 + stop_pct)
            for day in bars.close.index[bars.close.index > since_day]:
                o, h, l = bars.open.at[day, s], bars.high.at[day, s], bars.low.at[day, s]
                if q > 0 and l <= stop:
                    px = min(o, stop)
                elif q < 0 and h >= stop:
                    px = max(o, stop)
                else:
                    continue
                fee = self._fill(s, -q, px)
                fills.append({"symbol": s, "qty": abs(q), "price": px, "fee": fee,
                              "status": "FILLED", "reason": "stop", "day": str(day.date())})
                break
        return fills

    def flatten(self, prices: dict[str, float], tag: str = "") -> list[dict]:
        out = []
        for s, q in self.positions().items():
            o = Order(s, "SELL" if q > 0 else "BUY", Decimal(str(abs(q))), True, "flatten",
                      prices.get(s, self.book["positions"][s]["entry"]))
            out.append(self.execute(o, tag))
        return out


# =============================================================================== live
class LiveBroker:
    mode = "live"
    STOP_PREFIX = "v4s-"

    def __init__(self, client: BinanceFuturesClient, filters: dict[str, SymbolFilters],
                 margin_type: str, exchange_leverage: int, configured: dict, sleep=time.sleep):
        self.client, self.filters = client, filters
        self.margin_type, self.exchange_leverage = margin_type, exchange_leverage
        self.configured = configured             # persisted {symbol: "margin/leverage"} cache
        self.sleep = sleep
        self._pos_cache = None

    def prepare(self):
        self.client.sync_time()
        if self.client.dual_side_position():
            raise RuntimeError("account is in Hedge Mode; switch Binance futures to One-way Mode")

    def equity(self, prices=None) -> float:
        return float(self.client.account()["totalMarginBalance"])

    def _positions_raw(self):
        return [p for p in self.client.position_risk() if float(p.get("positionAmt", 0)) != 0]

    def positions(self) -> dict[str, float]:
        return {p["symbol"]: float(p["positionAmt"]) for p in self._positions_raw()}

    def entries(self) -> dict[str, float]:
        return {p["symbol"]: float(p["entryPrice"]) for p in self._positions_raw()}

    def _ensure_symbol(self, sym: str):
        key = f"{self.margin_type}/{self.exchange_leverage}"
        if self.configured.get(sym) == key:
            return
        try:
            self.client.set_margin_type(sym, self.margin_type)
        except BinanceAPIError as e:             # e.g. open position prevents changing it
            log.warning("%s: margin type not changed (%s)", sym, e)
        self.client.set_leverage(sym, self.exchange_leverage)
        self.configured[sym] = key

    def execute(self, order: Order, tag: str) -> dict:
        if not order.reduce_only:
            self._ensure_symbol(order.symbol)
        cid = f"v4-{tag}"[:36]
        params = dict(symbol=order.symbol, side=order.side, type="MARKET", quantity=fmt_dec(order.qty),
                      newClientOrderId=cid, newOrderRespType="RESULT")
        if order.reduce_only:
            params["reduceOnly"] = "true"
        try:
            r = self.client.new_order(**params)
        except AmbiguousOrderError as e:
            log.warning("%s: ambiguous order result (%s); checking by clientOrderId %s", order.symbol, e, cid)
            r = self._recover(order.symbol, cid, params)
        except BinanceAPIError as e:
            log.error("%s order rejected: %s", order.symbol, e)
            return {"symbol": order.symbol, "status": "REJECTED", "reason": str(e), "client_order_id": cid}
        return {"symbol": order.symbol, "side": order.side, "qty": float(r.get("executedQty", 0)),
                "price": float(r.get("avgPrice", 0) or 0), "status": r.get("status"),
                "reduce_only": order.reduce_only, "reason": order.reason, "client_order_id": cid,
                "order_id": r.get("orderId")}

    def _recover(self, symbol, cid, params):
        for attempt in range(3):
            self.sleep(2 * (attempt + 1))
            try:
                return self.client.query_order(symbol, cid)
            except BinanceAPIError as e:
                if e.code == -2013:              # order does not exist -> never reached the book
                    try:
                        return self.client.new_order(**params)
                    except AmbiguousOrderError:
                        continue
                raise
            except AmbiguousOrderError:
                continue
        raise RuntimeError(f"{symbol}: could not determine state of order {cid}; check the exchange")

    def place_stops(self, stop_pct: float, tag: str) -> list[dict]:
        """Replace this bot's catastrophe stops: STOP_MARKET closePosition at entry -/+ stop_pct."""
        out = []
        if stop_pct <= 0:
            return out
        for o in self.client.open_orders():
            if str(o.get("clientOrderId", "")).startswith(self.STOP_PREFIX):
                try:
                    self.client.cancel_order(o["symbol"], o["clientOrderId"])
                except BinanceAPIError as e:
                    log.warning("cancel stop %s failed: %s", o["clientOrderId"], e)
        for i, p in enumerate(self._positions_raw()):
            sym, q, e = p["symbol"], float(p["positionAmt"]), float(p["entryPrice"])
            f = self.filters.get(sym)
            if f is None or e <= 0:
                continue
            raw = Decimal(str(e * (1 - stop_pct) if q > 0 else e * (1 + stop_pct)))
            rnd = ROUND_DOWN if q > 0 else ROUND_UP
            stop = (raw / f.tick).to_integral_value(rounding=rnd) * f.tick
            cid = f"{self.STOP_PREFIX}{tag}-{i}"[:36]
            try:
                self.client.new_order(symbol=sym, side="SELL" if q > 0 else "BUY", type="STOP_MARKET",
                                      stopPrice=fmt_dec(stop), closePosition="true",
                                      workingType="MARK_PRICE", newClientOrderId=cid)
                out.append({"symbol": sym, "stop": float(stop)})
            except (BinanceAPIError, AmbiguousOrderError) as e:
                log.error("%s: failed to place stop: %s", sym, e)
                out.append({"symbol": sym, "error": str(e)})
        return out

    def flatten(self, prices=None, tag: str = "flat") -> list[dict]:
        out = []
        for i, (s, q) in enumerate(self.positions().items()):
            o = Order(s, "SELL" if q > 0 else "BUY", Decimal(str(abs(q))), True, "flatten")
            out.append(self.execute(o, f"{tag}-{i}"))
        return out
