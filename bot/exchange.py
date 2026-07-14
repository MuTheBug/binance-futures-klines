"""
Minimal, auditable Binance USDT-M Futures REST client.

Only depends on `requests`. Signs private calls with HMAC-SHA256. Supports MAINNET and
TESTNET (testnet.binancefuture.com). Every order goes through `place_order`, which honours
the bot's DRY_RUN flag — in dry-run it logs/returns the intended order WITHOUT sending it.

You can audit exactly which endpoints touch your account: account(), positions(),
set_leverage(), place_order(). Public market data (exchange_info, klines, prices) is unsigned.
"""
from __future__ import annotations
import hmac
import hashlib
import json
import time
import math
import urllib.parse
import requests

MAINNET = "https://fapi.binance.com"
TESTNET = "https://testnet.binancefuture.com"


class BinanceError(Exception):
    pass


class BinanceFutures:
    def __init__(self, api_key="", api_secret="", testnet=True, recv_window=5000, dry_run=True):
        self.key = api_key
        self.secret = api_secret.encode()
        self.base = TESTNET if testnet else MAINNET
        self.recv_window = recv_window
        self.dry_run = dry_run
        self.time_offset = 0          # serverTime - localTime (ms), set by sync_time()
        self.s = requests.Session()
        if api_key:
            self.s.headers.update({"X-MBX-APIKEY": api_key})
        self._filters = {}    # symbol -> filter dict (lazy)

    # ----------------------------- low-level -----------------------------
    def sync_time(self):
        """Align local clock to Binance server time (avoids -1021 timestamp errors)."""
        try:
            srv = self._request("GET", "/fapi/v1/time")["serverTime"]
            self.time_offset = int(srv) - int(time.time() * 1000)
        except Exception:
            self.time_offset = 0
        return self.time_offset

    def _request(self, method, path, params=None, signed=False):
        params = dict(params or {})
        url = self.base + path
        if signed:
            params["timestamp"] = int(time.time() * 1000) + self.time_offset
            params["recvWindow"] = self.recv_window
            query = urllib.parse.urlencode(params, doseq=True)
            sig = hmac.new(self.secret, query.encode(), hashlib.sha256).hexdigest()
            params["signature"] = sig
        try:
            r = self.s.request(method, url, params=params, timeout=20)
        except requests.RequestException as e:
            raise BinanceError(f"network error: {e}")
        if r.status_code != 200:
            raise BinanceError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    # ----------------------------- public -----------------------------
    def ping(self):
        return self._request("GET", "/fapi/v1/ping")

    def exchange_info(self):
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def klines(self, symbol, interval="1d", limit=300):
        return self._request("GET", "/fapi/v1/klines",
                             {"symbol": symbol, "interval": interval, "limit": limit})

    def top_long_short_position_ratio(self, symbol, period="1d", limit=30):
        """Top-trader long/short POSITION ratio (Sleeve C input). The /futures/data/*
        endpoints live on MAINNET only, so always read them from mainnet (public data),
        even when trading on testnet. Returns [(ms, ratio), ...] or [] if unavailable."""
        try:
            r = self.s.get(MAINNET + "/futures/data/topLongShortPositionRatio",
                           params={"symbol": symbol, "period": period, "limit": limit}, timeout=20)
            if r.status_code != 200:
                return []
            return [(int(d["timestamp"]), float(d["longShortRatio"])) for d in r.json()]
        except (requests.RequestException, ValueError, KeyError):
            return []

    def mark_prices(self):
        """All symbols' mark prices as {symbol: float}."""
        data = self._request("GET", "/fapi/v1/premiumIndex")
        return {d["symbol"]: float(d["markPrice"]) for d in data}

    def prices(self):
        data = self._request("GET", "/fapi/v1/ticker/price")
        return {d["symbol"]: float(d["price"]) for d in data}

    # ----------------------------- private -----------------------------
    def account(self):
        return self._request("GET", "/fapi/v2/account", signed=True)

    def hedge_mode(self):
        """True if account is in Hedge (dual-side) mode. This bot assumes ONE-WAY mode;
        hedge mode would make the one-way order logic incorrect, so the bot refuses to
        trade live until it's switched off (Binance app: Settings > Position Mode > One-way)."""
        return bool(self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
                    .get("dualSidePosition", False))

    def balance_usdt(self):
        """Total wallet equity in USDT (walletBalance + unrealized PnL)."""
        acc = self.account()
        return float(acc.get("totalMarginBalance", acc.get("totalWalletBalance", 0.0)))

    def positions(self):
        """Open positions as {symbol: signed_amount} (positive long, negative short)."""
        data = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        out = {}
        for p in data:
            amt = float(p["positionAmt"])
            if amt != 0.0:
                out[p["symbol"]] = amt
        return out

    def position_detail(self):
        """Open positions with entry price and unrealized PnL."""
        data = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        out = []
        for p in data:
            amt = float(p["positionAmt"])
            if amt != 0.0:
                out.append({
                    "symbol": p["symbol"], "amt": amt,
                    "entry": float(p["entryPrice"]), "mark": float(p["markPrice"]),
                    "upnl": float(p["unRealizedProfit"]),
                    "lev": float(p.get("leverage", 0) or 0),
                })
        return out

    def set_leverage(self, symbol, leverage):
        if self.dry_run:
            return {"dry_run": True, "symbol": symbol, "leverage": leverage}
        try:
            return self._request("POST", "/fapi/v1/leverage", signed=True,
                                 params={"symbol": symbol, "leverage": int(leverage)})
        except BinanceError as e:
            return {"error": str(e), "symbol": symbol}

    def book_ticker(self, symbol):
        """Best bid/ask for a symbol."""
        res = self._request("GET", "/fapi/v1/ticker/bookTicker",
                            params={"symbol": symbol})
        return float(res["bidPrice"]), float(res["askPrice"])

    def round_price(self, symbol, price):
        f = self._filters.get(symbol)
        if not f or f.get("tick", 0) <= 0:
            return float(price)
        tick = f["tick"]
        rounded = math.floor(float(price) / tick) * tick
        decimals = max(0, int(round(-math.log10(tick)))) if tick < 1 else 0
        return round(rounded, decimals)

    def place_post_only(self, symbol, side, quantity, price, reduce_only=False):
        """Post-only (GTX) limit order: joins the book as maker or is rejected."""
        params = {"symbol": symbol, "side": side, "type": "LIMIT",
                  "timeInForce": "GTX", "quantity": quantity, "price": price}
        if reduce_only:
            params["reduceOnly"] = "true"
        if self.dry_run:
            return {"dry_run": True, **params}
        return self._request("POST", "/fapi/v1/order", signed=True, params=params)

    def get_order(self, symbol, order_id):
        return self._request("GET", "/fapi/v1/order", signed=True,
                             params={"symbol": symbol, "orderId": order_id})

    def cancel_order(self, symbol, order_id):
        return self._request("DELETE", "/fapi/v1/order", signed=True,
                             params={"symbol": symbol, "orderId": order_id})

    def place_order(self, symbol, side, quantity, reduce_only=False, order_type="MARKET"):
        """Place a futures order. side in {BUY, SELL}, quantity > 0 (already rounded).
        Honours dry_run (returns the intended order without sending)."""
        params = {"symbol": symbol, "side": side, "type": order_type,
                  "quantity": quantity}
        if reduce_only:
            params["reduceOnly"] = "true"
        if self.dry_run:
            return {"dry_run": True, **params}
        return self._request("POST", "/fapi/v1/order", signed=True, params=params)

    # ----------------------------- filters / rounding -----------------------------
    def load_filters(self):
        """Map symbol -> {step, minQty, tick, minNotional} from exchangeInfo (PERPETUAL USDT)."""
        info = self.exchange_info()
        out = {}
        for s in info["symbols"]:
            if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT":
                continue
            if s.get("status") != "TRADING":
                continue
            f = {"step": 0.0, "minQty": 0.0, "tick": 0.0, "minNotional": 0.0,
                 "base": s["baseAsset"]}
            for flt in s["filters"]:
                if flt["filterType"] == "LOT_SIZE":
                    f["step"] = float(flt["stepSize"]); f["minQty"] = float(flt["minQty"])
                elif flt["filterType"] == "PRICE_FILTER":
                    f["tick"] = float(flt["tickSize"])
                elif flt["filterType"] in ("MIN_NOTIONAL", "NOTIONAL"):
                    f["minNotional"] = float(flt.get("notional", flt.get("minNotional", 0)) or 0)
            out[s["symbol"]] = f
        self._filters = out
        return out

    def round_qty(self, symbol, qty):
        f = self._filters.get(symbol)
        if not f or f["step"] <= 0:
            return float(qty)
        step = f["step"]
        rounded = math.floor(abs(qty) / step) * step
        # fix float dust
        decimals = max(0, int(round(-math.log10(step)))) if step < 1 else 0
        return round(rounded, decimals)


# ============================================================================
# Bybit V5 USDT-perp (linear) client — EXACT same interface as BinanceFutures,
# so signals.py / trader.py / bot.py are unchanged. Select it with EXCHANGE=bybit.
# ============================================================================
BYBIT_MAINNET = "https://api.bybit.com"
BYBIT_TESTNET = "https://api-testnet.bybit.com"

# bot only ever asks for "1d"; others mapped for completeness
_BYBIT_INTERVAL = {"1d": "D", "1h": "60", "4h": "240", "1w": "W", "1m": "1"}


class BybitFutures:
    """Bybit V5 linear-perp REST client mirroring BinanceFutures' interface.

    Signed calls use Bybit's V5 header scheme (X-BAPI-*): the signature is
    HMAC-SHA256 over (timestamp + api_key + recv_window + queryString|body).
    Order/side vocabulary is translated to Binance's (BUY/SELL, MARKET/LIMIT,
    status FILLED/…) so the rest of the bot needs no changes. Raises the same
    BinanceError used across the bot (generic "exchange error"). Sleeve-C
    positioning data is still sourced from Binance public data (the validated
    signal), independent of the execution venue. All orders pin positionIdx=0
    (one-way); a hedge-mode account would surface a clear error.
    """

    def __init__(self, api_key="", api_secret="", testnet=True, recv_window=5000, dry_run=True):
        self.key = api_key
        self.secret = api_secret.encode()
        self.base = BYBIT_TESTNET if testnet else BYBIT_MAINNET
        self.recv_window = recv_window
        self.dry_run = dry_run
        self.time_offset = 0
        self.s = requests.Session()
        self._filters = {}

    # ----------------------------- low-level -----------------------------
    def sync_time(self):
        try:
            r = self.s.get(self.base + "/v5/market/time", timeout=20).json()
            srv = int(int(r["result"]["timeNano"]) // 1_000_000)
            self.time_offset = srv - int(time.time() * 1000)
        except Exception:
            self.time_offset = 0
        return self.time_offset

    def _headers(self, ts, sign):
        return {"X-BAPI-API-KEY": self.key, "X-BAPI-TIMESTAMP": ts,
                "X-BAPI-RECV-WINDOW": str(self.recv_window), "X-BAPI-SIGN": sign}

    def _pub(self, path, params=None):
        try:
            r = self.s.get(self.base + path, params=params or {}, timeout=20)
        except requests.RequestException as e:
            raise BinanceError(f"network error: {e}")
        if r.status_code != 200:
            raise BinanceError(f"GET {path} -> HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        if j.get("retCode", 0) != 0:
            raise BinanceError(f"GET {path} -> {j.get('retCode')}: {j.get('retMsg')}")
        return j.get("result", {})

    def _signed(self, method, path, params=None):
        params = dict(params or {})
        ts = str(int(time.time() * 1000) + self.time_offset)
        recv = str(self.recv_window)
        if method == "GET":
            query = urllib.parse.urlencode(params, doseq=True)
            sign = hmac.new(self.secret, (ts + self.key + recv + query).encode(),
                            hashlib.sha256).hexdigest()
            url = self.base + path + (("?" + query) if query else "")
            try:
                r = self.s.get(url, headers=self._headers(ts, sign), timeout=20)
            except requests.RequestException as e:
                raise BinanceError(f"network error: {e}")
        else:
            body = json.dumps(params, separators=(",", ":"))
            sign = hmac.new(self.secret, (ts + self.key + recv + body).encode(),
                            hashlib.sha256).hexdigest()
            headers = self._headers(ts, sign)
            headers["Content-Type"] = "application/json"
            try:
                r = self.s.post(self.base + path, data=body, headers=headers, timeout=20)
            except requests.RequestException as e:
                raise BinanceError(f"network error: {e}")
        if r.status_code != 200:
            raise BinanceError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        if j.get("retCode", 0) != 0:
            raise BinanceError(f"{method} {path} -> {j.get('retCode')}: {j.get('retMsg')}")
        return j.get("result", {})

    # ----------------------------- public -----------------------------
    def ping(self):
        return self._pub("/v5/market/time")

    def klines(self, symbol, interval="1d", limit=300):
        """Binance-compatible rows [openTime, o, h, l, close, volume, closeTime]."""
        iv = _BYBIT_INTERVAL.get(interval, "D")
        res = self._pub("/v5/market/kline",
                        {"category": "linear", "symbol": symbol,
                         "interval": iv, "limit": min(int(limit), 1000)})
        span = 86_400_000 if iv == "D" else 3_600_000
        rows = []
        for k in res.get("list", []):
            ot = int(k[0])
            rows.append([ot, float(k[1]), float(k[2]), float(k[3]),
                         float(k[4]), float(k[5]), ot + span - 1])
        rows.sort(key=lambda x: x[0])          # Bybit returns newest-first
        return rows

    def top_long_short_position_ratio(self, symbol, period="1d", limit=30):
        """Sleeve-C input — sourced from Binance public data regardless of venue
        (this is the exact signal that was validated). Returns [] if unavailable."""
        try:
            r = self.s.get(MAINNET + "/futures/data/topLongShortPositionRatio",
                           params={"symbol": symbol, "period": period, "limit": limit}, timeout=20)
            if r.status_code != 200:
                return []
            return [(int(d["timestamp"]), float(d["longShortRatio"])) for d in r.json()]
        except (requests.RequestException, ValueError, KeyError):
            return []

    def prices(self):
        res = self._pub("/v5/market/tickers", {"category": "linear"})
        return {d["symbol"]: float(d["lastPrice"])
                for d in res.get("list", []) if d.get("lastPrice")}

    def mark_prices(self):
        res = self._pub("/v5/market/tickers", {"category": "linear"})
        return {d["symbol"]: float(d["markPrice"])
                for d in res.get("list", []) if d.get("markPrice")}

    def book_ticker(self, symbol):
        res = self._pub("/v5/market/tickers", {"category": "linear", "symbol": symbol})
        d = res["list"][0]
        return float(d["bid1Price"]), float(d["ask1Price"])

    # ----------------------------- private -----------------------------
    def account(self):
        return self._signed("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})

    def hedge_mode(self):
        """One-way is assumed (orders pin positionIdx=0). Report True only if a
        live position exposes a hedge-mode index, so the bot can refuse to trade."""
        try:
            res = self._signed("GET", "/v5/position/list",
                               {"category": "linear", "settleCoin": "USDT"})
            return any(int(p.get("positionIdx", 0)) != 0 for p in res.get("list", []))
        except BinanceError:
            return False

    def balance_usdt(self):
        lst = self.account().get("list", [])
        if not lst:
            return 0.0
        a = lst[0]
        return float(a.get("totalMarginBalance") or a.get("totalEquity") or 0.0)

    def positions(self):
        res = self._signed("GET", "/v5/position/list",
                           {"category": "linear", "settleCoin": "USDT"})
        out = {}
        for p in res.get("list", []):
            size = float(p.get("size", 0) or 0)
            if size != 0.0:
                out[p["symbol"]] = size if p.get("side") == "Buy" else -size
        return out

    def position_detail(self):
        res = self._signed("GET", "/v5/position/list",
                           {"category": "linear", "settleCoin": "USDT"})
        out = []
        for p in res.get("list", []):
            size = float(p.get("size", 0) or 0)
            if size == 0.0:
                continue
            out.append({
                "symbol": p["symbol"],
                "amt": size if p.get("side") == "Buy" else -size,
                "entry": float(p.get("avgPrice") or 0),
                "mark": float(p.get("markPrice") or 0),
                "upnl": float(p.get("unrealisedPnl") or 0),
                "lev": float(p.get("leverage") or 0),
            })
        return out

    def set_leverage(self, symbol, leverage):
        if self.dry_run:
            return {"dry_run": True, "symbol": symbol, "leverage": leverage}
        try:
            return self._signed("POST", "/v5/position/set-leverage",
                                {"category": "linear", "symbol": symbol,
                                 "buyLeverage": str(int(leverage)),
                                 "sellLeverage": str(int(leverage))})
        except BinanceError as e:
            if "110043" in str(e):     # leverage not modified (already set) -> benign
                return {"symbol": symbol, "leverage": leverage, "unchanged": True}
            return {"error": str(e), "symbol": symbol}

    @staticmethod
    def _side(side):
        return "Buy" if str(side).upper() == "BUY" else "Sell"

    def book_order_params(self, symbol, side, quantity, reduce_only, order_type, price=None):
        p = {"category": "linear", "symbol": symbol, "side": self._side(side),
             "orderType": "Market" if order_type.upper() == "MARKET" else "Limit",
             "qty": str(quantity), "positionIdx": 0}
        if price is not None:
            p["price"] = str(price)
        if reduce_only:
            p["reduceOnly"] = True
        return p

    def place_order(self, symbol, side, quantity, reduce_only=False, order_type="MARKET"):
        params = self.book_order_params(symbol, side, quantity, reduce_only, order_type)
        if self.dry_run:
            return {"dry_run": True, **params}
        return self._signed("POST", "/v5/order/create", params)

    def place_post_only(self, symbol, side, quantity, price, reduce_only=False):
        params = self.book_order_params(symbol, side, quantity, reduce_only, "LIMIT", price)
        params["timeInForce"] = "PostOnly"
        if self.dry_run:
            return {"dry_run": True, **params}
        return self._signed("POST", "/v5/order/create", params)

    _STATUS = {"Filled": "FILLED", "New": "NEW", "PartiallyFilled": "NEW",
               "Cancelled": "CANCELED", "Rejected": "REJECTED",
               "Deactivated": "EXPIRED", "Untriggered": "NEW"}

    def get_order(self, symbol, order_id):
        res = self._signed("GET", "/v5/order/realtime",
                           {"category": "linear", "symbol": symbol, "orderId": order_id})
        lst = res.get("list", [])
        if not lst:
            return {"status": "CANCELED", "executedQty": 0.0}
        o = lst[0]
        return {"status": self._STATUS.get(o.get("orderStatus"), "NEW"),
                "executedQty": float(o.get("cumExecQty") or 0),
                "orderId": o.get("orderId")}

    def cancel_order(self, symbol, order_id):
        return self._signed("POST", "/v5/order/cancel",
                            {"category": "linear", "symbol": symbol, "orderId": order_id})

    # ----------------------------- filters / rounding -----------------------------
    def load_filters(self):
        """symbol -> {step, minQty, tick, minNotional, base} for linear USDT perps."""
        res = self._pub("/v5/market/instruments-info", {"category": "linear"})
        out = {}
        for s in res.get("list", []):
            if s.get("quoteCoin") != "USDT" or s.get("status") != "Trading":
                continue
            if s.get("contractType") not in ("LinearPerpetual", None):
                continue
            lot = s.get("lotSizeFilter", {})
            pf = s.get("priceFilter", {})
            out[s["symbol"]] = {
                "step": float(lot.get("qtyStep") or 0),
                "minQty": float(lot.get("minOrderQty") or 0),
                "tick": float(pf.get("tickSize") or 0),
                "minNotional": float(lot.get("minNotionalValue") or 0),
                "base": s.get("baseCoin", s["symbol"].replace("USDT", "")),
            }
        self._filters = out
        return out

    # identical rounding logic as Binance (pure functions of self._filters)
    round_price = BinanceFutures.round_price
    round_qty = BinanceFutures.round_qty


def make_exchange(cfg):
    """Instantiate the exchange client selected by cfg.exchange ('binance' or
    'bybit'). Both expose the identical interface, so the rest of the bot is
    unchanged; Binance remains the default."""
    name = (getattr(cfg, "exchange", "binance") or "binance").strip().lower()
    klass = BybitFutures if name == "bybit" else BinanceFutures
    return klass(cfg.api_key, cfg.api_secret, testnet=cfg.testnet,
                 recv_window=cfg.recv_window, dry_run=cfg.dry_run)
