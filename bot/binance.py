"""Minimal, dependency-light Binance USDT-M futures REST client.

Handles HMAC-SHA256 signing, server-time sync, request-weight throttling, and retries:
  * GET/DELETE are retried on network errors, HTTP 5xx and 429/418 (with Retry-After).
  * POST (orders) is NOT blindly retried: an ambiguous failure raises AmbiguousOrderError so the
    caller can look the order up by its clientOrderId (idempotent order placement).
"""
from __future__ import annotations
import hashlib
import hmac
import logging
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from urllib.parse import urlencode

import requests

log = logging.getLogger(__name__)

MAINNET = "https://fapi.binance.com"
TESTNET = "https://testnet.binancefuture.com"


class BinanceAPIError(Exception):
    def __init__(self, status: int, code: int | None, msg: str):
        super().__init__(f"HTTP {status} code={code} {msg}")
        self.status, self.code, self.msg = status, code, msg


class AmbiguousOrderError(Exception):
    """The order request may or may not have reached the exchange."""


@dataclass(frozen=True)
class SymbolFilters:
    symbol: str
    base: str
    step: Decimal          # MARKET_LOT_SIZE (falls back to LOT_SIZE) step
    min_qty: Decimal
    max_qty: Decimal
    tick: Decimal
    min_notional: float


class BinanceFuturesClient:
    WEIGHT_LIMIT = 2400

    def __init__(self, base_url: str = MAINNET, api_key: str = "", api_secret: str = "",
                 recv_window: int = 5000, session=None, timeout: float = 15.0,
                 max_retries: int = 4, sleep=time.sleep, clock=time.time):
        self.base_url = base_url.rstrip("/")
        self.api_key, self._secret = api_key, api_secret.encode()
        self.recv_window = recv_window
        self.session = session or requests.Session()
        self.timeout, self.max_retries = timeout, max_retries
        self.sleep, self.clock = sleep, clock
        self.time_offset_ms = 0
        self._lock = threading.Lock()
        self.used_weight = 0

    # ------------------------------------------------------------------ core
    def sign(self, query: str) -> str:
        return hmac.new(self._secret, query.encode(), hashlib.sha256).hexdigest()

    def _throttle(self, headers):
        w = headers.get("X-MBX-USED-WEIGHT-1M") or headers.get("x-mbx-used-weight-1m")
        if w is None:
            return
        with self._lock:
            self.used_weight = int(w)
        if self.used_weight > 0.8 * self.WEIGHT_LIMIT:
            wait = 61 - (self.clock() % 60)
            log.warning("request weight %d near limit; sleeping %.0fs", self.used_weight, wait)
            self.sleep(wait)

    def request(self, method: str, path: str, params: dict | None = None, signed: bool = False):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}
        resynced = False
        for attempt in range(self.max_retries + 1):
            p = dict(params)
            if signed:
                if not self._secret:
                    raise RuntimeError("signed endpoint requires API secret")
                p["recvWindow"] = self.recv_window
                p["timestamp"] = int(self.clock() * 1000) + self.time_offset_ms
                q = urlencode(p)
                q += "&signature=" + self.sign(q)
            else:
                q = urlencode(p)
            url = f"{self.base_url}{path}" + (f"?{q}" if q else "")
            try:
                resp = self.session.request(method, url, headers=headers, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                if method == "POST":
                    raise AmbiguousOrderError(str(e)) from e
                self._backoff(attempt, f"network error {e!r}")
                continue
            self._throttle(resp.headers)
            if resp.status_code in (418, 429):
                wait = float(resp.headers.get("Retry-After", 2 ** attempt * 5))
                log.warning("rate limited (HTTP %s); sleeping %.0fs", resp.status_code, wait)
                self.sleep(wait)
                continue
            if resp.status_code >= 500:
                if method == "POST":
                    raise AmbiguousOrderError(f"HTTP {resp.status_code}")
                self._backoff(attempt, f"HTTP {resp.status_code}")
                continue
            try:
                body = resp.json()
            except ValueError:
                body = {"msg": getattr(resp, "text", "")}
            if resp.status_code >= 400:
                code = body.get("code") if isinstance(body, dict) else None
                msg = body.get("msg", "") if isinstance(body, dict) else str(body)
                if code == -1021 and not resynced:        # timestamp outside recvWindow
                    self.sync_time(); resynced = True
                    continue
                raise BinanceAPIError(resp.status_code, code, msg)
            return body
        raise BinanceAPIError(0, None, f"{method} {path}: retries exhausted")

    def _backoff(self, attempt, why):
        if attempt >= self.max_retries:
            return
        wait = min(60.0, 2.0 ** attempt)
        log.warning("%s; retry %d in %.0fs", why, attempt + 1, wait)
        self.sleep(wait)

    # ------------------------------------------------------------------ public
    def sync_time(self):
        server = self.request("GET", "/fapi/v1/time")["serverTime"]
        self.time_offset_ms = int(server - self.clock() * 1000)
        return self.time_offset_ms

    def exchange_info(self):
        return self.request("GET", "/fapi/v1/exchangeInfo")

    def klines(self, symbol: str, interval: str = "1d", limit: int = 500, end_time: int | None = None):
        return self.request("GET", "/fapi/v1/klines",
                            {"symbol": symbol, "interval": interval, "limit": limit, "endTime": end_time})

    def ticker_24h(self):
        return self.request("GET", "/fapi/v1/ticker/24hr")

    def premium_index(self):
        return self.request("GET", "/fapi/v1/premiumIndex")

    def funding_rates(self, symbol: str, start_ms: int, end_ms: int):
        return self.request("GET", "/fapi/v1/fundingRate",
                            {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000})

    # ------------------------------------------------------------------ signed
    def account(self):
        return self.request("GET", "/fapi/v3/account", signed=True)

    def position_risk(self):
        return self.request("GET", "/fapi/v3/positionRisk", signed=True)

    def dual_side_position(self) -> bool:
        return bool(self.request("GET", "/fapi/v1/positionSide/dual", signed=True)["dualSidePosition"])

    def set_leverage(self, symbol: str, leverage: int):
        return self.request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    def set_margin_type(self, symbol: str, margin_type: str):
        try:
            return self.request("POST", "/fapi/v1/marginType",
                                {"symbol": symbol, "marginType": margin_type}, signed=True)
        except BinanceAPIError as e:
            if e.code == -4046:          # "No need to change margin type."
                return None
            raise

    def new_order(self, **params):
        return self.request("POST", "/fapi/v1/order", params, signed=True)

    def query_order(self, symbol: str, client_order_id: str):
        return self.request("GET", "/fapi/v1/order",
                            {"symbol": symbol, "origClientOrderId": client_order_id}, signed=True)

    def open_orders(self, symbol: str | None = None):
        return self.request("GET", "/fapi/v1/openOrders", {"symbol": symbol}, signed=True)

    def cancel_order(self, symbol: str, client_order_id: str):
        return self.request("DELETE", "/fapi/v1/order",
                            {"symbol": symbol, "origClientOrderId": client_order_id}, signed=True)


def parse_filters(exchange_info: dict, exclude=()) -> dict[str, SymbolFilters]:
    """Tradable USDT-margined crypto perpetuals -> their order filters."""
    excl = set(exclude)
    out = {}
    for s in exchange_info.get("symbols", []):
        if (s.get("status") != "TRADING" or s.get("contractType") != "PERPETUAL"
                or s.get("quoteAsset") != "USDT" or s.get("marginAsset", "USDT") != "USDT"):
            continue
        base = s["baseAsset"]
        if base in excl or s.get("underlyingType", "COIN") not in ("COIN",):
            continue
        f = {x["filterType"]: x for x in s.get("filters", [])}
        lot = f.get("MARKET_LOT_SIZE") or f.get("LOT_SIZE")
        if not lot or Decimal(lot["stepSize"]) <= 0:
            lot = f.get("LOT_SIZE")
        if not lot:
            continue
        price = f.get("PRICE_FILTER", {"tickSize": "0.00000001"})
        notional = f.get("MIN_NOTIONAL", {}).get("notional", "5")
        out[s["symbol"]] = SymbolFilters(
            symbol=s["symbol"], base=base,
            step=Decimal(lot["stepSize"]), min_qty=Decimal(lot["minQty"]), max_qty=Decimal(lot["maxQty"]),
            tick=Decimal(price["tickSize"]), min_notional=float(notional))
    return out
