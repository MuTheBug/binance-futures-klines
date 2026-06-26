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
        self.s = requests.Session()
        if api_key:
            self.s.headers.update({"X-MBX-APIKEY": api_key})
        self._filters = {}    # symbol -> filter dict (lazy)

    # ----------------------------- low-level -----------------------------
    def _request(self, method, path, params=None, signed=False):
        params = dict(params or {})
        url = self.base + path
        if signed:
            params["timestamp"] = int(time.time() * 1000)
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
