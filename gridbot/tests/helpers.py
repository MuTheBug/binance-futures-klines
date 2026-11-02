"""Test helpers: synthetic bars, and a fake Binance HTTP server backed by SimExchange so the real
BinanceFuturesClient (signing, params, error mapping) can be exercised end to end."""
from __future__ import annotations
import hashlib
import hmac
from decimal import Decimal
from urllib.parse import parse_qsl, urlsplit

import numpy as np
import pandas as pd

from bot.binance import BinanceAPIError, SymbolFilters


def random_walk_bars(n_hours=1500, seed=3, vol=0.01, start=100.0, drift=0.0):
    rng = np.random.default_rng(seed)
    m = start * np.exp(np.cumsum(rng.normal(drift / 60, vol / np.sqrt(60), n_hours * 60)))
    mm = m.reshape(n_hours, 60)
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    return pd.DataFrame({"open": mm[:, 0], "high": mm.max(1), "low": mm.min(1), "close": mm[:, -1]}, index=idx)


def filters_for(symbols, step="0.001", tick="0.0001", min_notional=5.0):
    return {s: SymbolFilters(s, s[:-4], Decimal(step), Decimal(step), Decimal("1e9"), Decimal(tick), min_notional)
            for s in symbols}


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.headers = {"X-MBX-USED-WEIGHT-1M": "5"}
        self.text = str(body)

    def json(self):
        return self._body


class FakeBinanceHTTP:
    """requests.Session stand-in: routes signed/public REST calls to a SimExchange."""

    def __init__(self, sim, key="k", secret="s", klines=None):
        self.sim, self.key, self.secret = sim, key, secret
        self.klines = klines or {}                 # (symbol, interval) -> rows
        self.calls = []

    def request(self, method, url, headers=None, timeout=None):
        u = urlsplit(url)
        p = dict(parse_qsl(u.query, keep_blank_values=True))
        self.calls.append((method, u.path, p))
        if "signature" in p:
            sig = p.pop("signature")
            q = u.query.rsplit("&signature=", 1)[0]
            if hmac.new(self.secret.encode(), q.encode(), hashlib.sha256).hexdigest() != sig:
                return Resp(400, {"code": -1022, "msg": "bad signature"})
            p.pop("timestamp"); p.pop("recvWindow")
        try:
            return Resp(200, self.route(method, u.path, p))
        except BinanceAPIError as e:
            return Resp(400, {"code": e.code, "msg": e.msg})

    def route(self, m, path, p):
        sim = self.sim
        if (m, path) == ("GET", "/fapi/v1/time"):
            return {"serverTime": 0}
        if (m, path) == ("GET", "/fapi/v1/exchangeInfo"):
            return {"symbols": [{"symbol": s, "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
                                 "marginAsset": "USDT", "baseAsset": f.base, "underlyingType": "COIN",
                                 "filters": [{"filterType": "LOT_SIZE", "stepSize": str(f.step), "minQty": str(f.min_qty),
                                              "maxQty": str(f.max_qty)},
                                             {"filterType": "PRICE_FILTER", "tickSize": str(f.tick)},
                                             {"filterType": "MIN_NOTIONAL", "notional": str(f.min_notional)}]}
                                for s, f in sim.filters.items()]}
        if (m, path) == ("GET", "/fapi/v1/ticker/price"):
            return [{"symbol": s, "price": str(px)} for s, px in sim.price.items()]
        if (m, path) == ("GET", "/fapi/v1/premiumIndex"):
            return sim.premium_index()
        if (m, path) == ("GET", "/fapi/v1/klines"):
            return self.klines.get((p["symbol"], p["interval"]), [])[-int(p.get("limit", 500)):]
        if (m, path) == ("GET", "/fapi/v3/account"):
            return sim.account()
        if (m, path) == ("GET", "/fapi/v3/positionRisk"):
            return sim.position_risk()
        if (m, path) == ("GET", "/fapi/v1/positionSide/dual"):
            return {"dualSidePosition": False}
        if (m, path) == ("POST", "/fapi/v1/leverage"):
            return sim.set_leverage(p["symbol"], int(p["leverage"]))
        if (m, path) == ("POST", "/fapi/v1/marginType"):
            return {"code": 200}
        if (m, path) == ("GET", "/fapi/v1/openOrders"):
            return sim.open_orders(p.get("symbol"))
        if (m, path) == ("POST", "/fapi/v1/order"):
            return sim.new_order(**p)
        if (m, path) == ("DELETE", "/fapi/v1/order"):
            return sim.cancel_order(p["symbol"], p["origClientOrderId"])
        raise BinanceAPIError(404, -1, f"no route {m} {path}")
