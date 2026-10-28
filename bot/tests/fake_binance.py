"""A fake Binance USDT-M futures HTTP server (in-process `requests.Session` stand-in).

Serves exchangeInfo / 24h tickers / klines / premiumIndex / funding from the repo CSV data as of a
fixed `now`, and implements a small signed account: orders fill at the mark price, positions and
wallet are tracked, signatures are verified, clientOrderIds are unique.
"""
from __future__ import annotations
import hashlib
import hmac
from urllib.parse import parse_qsl, urlsplit

import pandas as pd


class Resp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body = status, body
        self.headers = {"X-MBX-USED-WEIGHT-1M": "10", **(headers or {})}
        self.text = str(body)

    def json(self):
        return self._body


class FakeBinance:
    def __init__(self, csv_data, now: pd.Timestamp, key="k", secret="s", wallet=10_000.0, fee=0.0005):
        self.data, self.now = csv_data, now
        self.key, self.secret = key, secret
        self.wallet, self.fee = wallet, fee
        self.positions: dict[str, float] = {}
        self.entry: dict[str, float] = {}
        self.orders: list[dict] = []
        self.client_ids: dict[str, dict] = {}
        self.open_stops: dict[str, dict] = {}
        self.leverage, self.margin = {}, {}
        self.dual = False
        self.fail_next_post = 0          # simulate timeouts AFTER the order reached the book
        self.requests: list[tuple] = []
        self.marks_override: dict[str, float] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def day(self):
        return self.now.floor("D") - pd.Timedelta(days=1)

    def marks(self):
        row = self.data.all.close.loc[:self.day].ffill().iloc[-1].dropna()
        m = row.to_dict()
        m.update(self.marks_override)
        return m

    def equity(self):
        m = self.marks()
        return self.wallet + sum(q * (m[s] - self.entry[s]) for s, q in self.positions.items())

    # ------------------------------------------------------------------ transport
    def request(self, method, url, headers=None, timeout=None):
        u = urlsplit(url)
        params = dict(parse_qsl(u.query, keep_blank_values=True))
        self.requests.append((method, u.path, params))
        if "signature" in params:
            sig = params.pop("signature")
            q = u.query.rsplit("&signature=", 1)[0]
            if hmac.new(self.secret.encode(), q.encode(), hashlib.sha256).hexdigest() != sig:
                return Resp(400, {"code": -1022, "msg": "Signature for this request is not valid."})
            if (headers or {}).get("X-MBX-APIKEY") != self.key:
                return Resp(401, {"code": -2015, "msg": "Invalid API-key"})
        h = getattr(self, "_" + method.lower() + u.path.replace("/", "_"), None)
        if h is None:
            return Resp(404, {"code": -1, "msg": f"no route {method} {u.path}"})
        return h(params)

    # ------------------------------------------------------------------ public
    def _get_fapi_v1_time(self, p):
        return Resp(200, {"serverTime": int(self.now.timestamp() * 1000)})

    def _get_fapi_v1_exchangeInfo(self, p):
        syms = []
        last = self.data.all.close.loc[:self.day].iloc[-1]
        for s in self.data.all.close.columns:
            if pd.isna(last.get(s)):
                continue
            syms.append({"symbol": s, "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
                         "marginAsset": "USDT", "baseAsset": s[:-4], "underlyingType": "COIN",
                         "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.0001"},
                                     {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000000000"},
                                     {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000000000"},
                                     {"filterType": "MIN_NOTIONAL", "notional": "5"}]})
        return Resp(200, {"symbols": syms})

    def _get_fapi_v1_ticker_24hr(self, p):
        c, v = self.data.all.close.loc[:self.day].iloc[-1], self.data.all.volume.loc[:self.day].iloc[-1]
        return Resp(200, [{"symbol": s, "quoteVolume": str(float(c[s] * v[s]))} for s in c.dropna().index])

    def _get_fapi_v1_klines(self, p):
        s, limit = p["symbol"], int(p.get("limit", 500))
        a = self.data.all
        # include the still-forming current-day bar, like the real API does
        idx = a.close.loc[:self.now.floor("D"), s].dropna().index[-limit:]
        rows = []
        for t in idx:
            ot = int(t.timestamp() * 1000)
            rows.append([ot, str(a.open.at[t, s]), str(a.high.at[t, s]), str(a.low.at[t, s]),
                         str(a.close.at[t, s]), str(a.volume.at[t, s]), ot + 86_399_999, "0", 0, "0", "0", "0"])
        return Resp(200, rows)

    def _get_fapi_v1_premiumIndex(self, p):
        return Resp(200, [{"symbol": s, "markPrice": str(px)} for s, px in self.marks().items()])

    def _get_fapi_v1_fundingRate(self, p):
        return Resp(200, [{"symbol": p["symbol"], "fundingRate": "0.0001", "fundingTime": int(p["endTime"])}])

    # ------------------------------------------------------------------ signed
    def _get_fapi_v3_account(self, p):
        return Resp(200, {"totalMarginBalance": str(self.equity()), "totalWalletBalance": str(self.wallet)})

    def _get_fapi_v3_positionRisk(self, p):
        m = self.marks()
        return Resp(200, [{"symbol": s, "positionAmt": str(q), "entryPrice": str(self.entry[s]),
                           "markPrice": str(m[s])} for s, q in self.positions.items() if q])

    def _get_fapi_v1_positionSide_dual(self, p):
        return Resp(200, {"dualSidePosition": self.dual})

    def _post_fapi_v1_leverage(self, p):
        self.leverage[p["symbol"]] = int(p["leverage"]); return Resp(200, {"leverage": int(p["leverage"])})

    def _post_fapi_v1_marginType(self, p):
        if self.margin.get(p["symbol"]) == p["marginType"]:
            return Resp(400, {"code": -4046, "msg": "No need to change margin type."})
        self.margin[p["symbol"]] = p["marginType"]; return Resp(200, {"code": 200, "msg": "success"})

    def _post_fapi_v1_order(self, p):
        cid = p["newClientOrderId"]
        if cid in self.client_ids:
            return Resp(400, {"code": -4116, "msg": "ClientOrderId is duplicated."})
        if p["type"] == "STOP_MARKET":
            o = {**p, "status": "NEW", "clientOrderId": cid}
            self.open_stops[cid] = o; self.client_ids[cid] = o
            return Resp(200, o)
        s, side, qty = p["symbol"], p["side"], float(p["quantity"])
        px = self.marks()[s]
        cur = self.positions.get(s, 0.0)
        dq = qty if side == "BUY" else -qty
        if p.get("reduceOnly") == "true":
            if cur == 0 or (cur > 0) == (dq > 0):
                return Resp(400, {"code": -2022, "msg": "ReduceOnly Order is rejected."})
            dq = min(abs(dq), abs(cur)) * (1 if dq > 0 else -1)
        elif abs(dq) * px < 5:
            return Resp(400, {"code": -4164, "msg": "Order's notional must be no smaller than 5"})
        self._apply(s, dq, px)
        o = {"orderId": len(self.orders) + 1, "symbol": s, "status": "FILLED", "clientOrderId": cid,
             "executedQty": str(abs(dq)), "avgPrice": str(px), "side": side, "reduceOnly": p.get("reduceOnly") == "true"}
        self.orders.append(o); self.client_ids[cid] = o
        if self.fail_next_post:
            self.fail_next_post -= 1
            return Resp(503, {"code": -1001, "msg": "Internal error; unable to process your request."})
        return Resp(200, o)

    def _apply(self, s, dq, px):
        q0, e0 = self.positions.get(s, 0.0), self.entry.get(s, 0.0)
        if q0 == 0 or (q0 > 0) == (dq > 0):
            q1 = q0 + dq; self.entry[s] = (q0 * e0 + dq * px) / q1
        else:
            closing = min(abs(dq), abs(q0))
            self.wallet += closing * (px - e0) * (1 if q0 > 0 else -1)
            q1 = q0 + dq
            if abs(q1) > 1e-12 and (q1 > 0) != (q0 > 0):
                self.entry[s] = px
        self.wallet -= abs(dq) * px * self.fee
        if abs(q1) < 1e-12:
            self.positions.pop(s, None); self.entry.pop(s, None)
        else:
            self.positions[s] = q1

    def _get_fapi_v1_order(self, p):
        o = self.client_ids.get(p["origClientOrderId"])
        if o is None:
            return Resp(400, {"code": -2013, "msg": "Order does not exist."})
        return Resp(200, o)

    def _get_fapi_v1_openOrders(self, p):
        return Resp(200, list(self.open_stops.values()))

    def _delete_fapi_v1_order(self, p):
        o = self.open_stops.pop(p["origClientOrderId"], None)
        return Resp(200, o) if o else Resp(400, {"code": -2011, "msg": "Unknown order sent."})
