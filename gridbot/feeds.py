"""Price feeds.

LivePrices  : last traded prices from Binance (live mode; the exchange does the matching).
PaperFeed   : drives a SimExchange from live public 1-minute klines. Every poll it replays, in
              order, the parts of each minute's path not yet seen (new lows/highs, then the latest
              price), so wicks between polls still fill resting paper orders. Applies the real
              funding rate at each 8-hour funding time.
"""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)
FUNDING_PERIOD_MS = 8 * 3600 * 1000


class LivePrices:
    def __init__(self, client, symbols):
        self.client, self.symbols = client, list(symbols)

    def update(self, now_ms: int) -> dict[str, float]:
        px = {p["symbol"]: float(p["price"]) for p in self.client.ticker_price()}
        return {s: px[s] for s in self.symbols if s in px}


class PaperFeed:
    def __init__(self, client, sim, symbols, state: dict):
        self.client, self.sim, self.symbols = client, sim, list(symbols)
        self.st = state.setdefault("paper_feed", {})     # symbol -> {minute, hi, lo}

    def _walk(self, sym, pts):
        for p in pts:
            if p and p > 0:
                self.sim.set_price(sym, p)

    def update(self, now_ms: int) -> dict[str, float]:
        for s in self.symbols:
            rows = self.client.klines(s, "1m", limit=3)
            seen = self.st.get(s)
            for k in rows:
                t, o, h, l, c = int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])
                if seen and t < seen["minute"]:
                    continue
                if seen is None or s not in self.sim.price:      # first ever price: just set it
                    self.sim.set_price(s, c)
                elif t > seen["minute"]:                          # a minute we have not seen at all
                    self._walk(s, [o, l, h, c] if c >= o else [o, h, l, c])
                else:                                             # same minute: only what is new
                    new_lo, new_hi = l < seen["lo"], h > seen["hi"]
                    path = []
                    if new_lo and new_hi:
                        path = [l, h] if c >= o else [h, l]
                    elif new_lo:
                        path = [l]
                    elif new_hi:
                        path = [h]
                    self._walk(s, path + [c])
                seen = {"minute": t, "hi": h if not seen or t > seen["minute"] else max(h, seen["hi"]),
                        "lo": l if not seen or t > seen["minute"] else min(l, seen["lo"])}
                self.st[s] = seen
        last_fund = self.st.get("_last_funding_ms", now_ms - now_ms % FUNDING_PERIOD_MS)
        if now_ms // FUNDING_PERIOD_MS > last_fund // FUNDING_PERIOD_MS:
            rates = {p["symbol"]: float(p.get("lastFundingRate", 0) or 0) for p in self.client.premium_index()}
            for s in self.symbols:
                self.sim.apply_funding(s, rates.get(s, 0.0))
            log.info("paper funding applied: %s", {s: rates.get(s) for s in self.symbols})
        self.st["_last_funding_ms"] = now_ms
        return {s: self.sim.price[s] for s in self.symbols if s in self.sim.price}
