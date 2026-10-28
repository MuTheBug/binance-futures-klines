import hashlib
import hmac
import os
import tempfile
import unittest
from decimal import Decimal

import pandas as pd
import requests

from bot.binance import AmbiguousOrderError, BinanceAPIError, BinanceFuturesClient, SymbolFilters, parse_filters
from bot.brokers import PaperBroker, fmt_dec
from bot.config import Config, RiskCfg, load_config
from bot.execution import Order, plan_orders, round_qty
from bot.runner import check_risk
from bot.tests.fake_binance import Resp


def filt(sym, step="0.001", min_notional=5.0):
    return SymbolFilters(sym, sym[:-4], Decimal(step), Decimal(step), Decimal("1e9"), Decimal("0.01"), min_notional)


class TestConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = load_config(None, env={})
        self.assertEqual(cfg.mode, "paper")

    def test_unknown_key_and_secrets_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write('[risk]\nleverag = 2\n')
        with self.assertRaisesRegex(ValueError, "unknown config key"):
            load_config(f.name, env={})
        with open(f.name, "w") as g:
            g.write('api_key = "x"\n')
        with self.assertRaisesRegex(ValueError, "environment"):
            load_config(f.name, env={})
        os.unlink(f.name)

    def test_live_needs_keys_and_leverage_cap(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write('mode = "live"\n')
        with self.assertRaisesRegex(ValueError, "BINANCE_API_KEY"):
            load_config(f.name, env={})
        self.assertEqual(load_config(f.name, env={"BINANCE_API_KEY": "a", "BINANCE_API_SECRET": "b"}).mode, "live")
        with open(f.name, "w") as g:
            g.write('[risk]\nleverage = 8.0\n')
        with self.assertRaisesRegex(ValueError, "leverage"):
            load_config(f.name, env={})
        os.unlink(f.name)


class TestExecution(unittest.TestCase):
    F = {s: filt(s) for s in ("AUSDT", "BUSDT", "CUSDT", "DUSDT")}
    P = {"AUSDT": 10.0, "BUSDT": 100.0, "CUSDT": 1.0, "DUSDT": 2.0}

    def test_round_toward_zero(self):
        self.assertEqual(round_qty(1.23456, Decimal("0.01")), Decimal("1.23"))
        self.assertEqual(round_qty(-1.23956, Decimal("0.01")), Decimal("-1.23"))

    def test_plan_basic_and_order(self):
        risk = RiskCfg(leverage=1.0, rebalance_band=0.0)
        w = pd.Series({"AUSDT": 0.5, "BUSDT": -0.3})
        pos = {"CUSDT": 100.0, "BUSDT": 2.0}              # close C, flip B
        plan = plan_orders(w, 1000.0, pos, self.P, self.F, risk)
        kinds = [(o.symbol, o.side, o.reduce_only, o.reason) for o in plan.orders]
        self.assertEqual(kinds[:2], [("BUSDT", "SELL", True, "flip-close"), ("CUSDT", "SELL", True, "close")])
        self.assertIn(("AUSDT", "BUY", False, "open"), kinds)
        self.assertIn(("BUSDT", "SELL", False, "flip-open"), kinds)
        self.assertTrue(all(o.reduce_only for o in plan.orders[:2]) and not any(o.reduce_only for o in plan.orders[2:]))
        self.assertAlmostEqual(plan.target_qty["AUSDT"], 50.0)
        self.assertAlmostEqual(plan.target_qty["BUSDT"], -3.0)

    def test_caps_band_and_min_notional(self):
        risk = RiskCfg(leverage=2.0, max_gross=1.0, max_position=0.4, rebalance_band=0.05)
        w = pd.Series({"AUSDT": 0.6, "BUSDT": 0.4, "CUSDT": 0.001})
        plan = plan_orders(w, 1000.0, {"BUSDT": 3.7}, self.P, self.F, risk)
        # A: 1200 -> cap 400; B: 800 -> cap 400; gross 800 (+C 2) <= 1000 -> no scale. C 2 USDT < 5 min notional -> 0
        self.assertAlmostEqual(plan.target_notional["AUSDT"], 400.0)
        self.assertEqual(plan.target_qty["CUSDT"], 0.0)
        # B currently 370 vs target 400: change 30 < band 50 -> skipped
        self.assertTrue(any(s.startswith("BUSDT") for s in plan.skipped))
        self.assertEqual([o.symbol for o in plan.orders], ["AUSDT"])
        risk2 = RiskCfg(leverage=2.0, max_gross=0.5, max_position=1.0, rebalance_band=0.0)
        plan2 = plan_orders(pd.Series({"AUSDT": 0.5, "BUSDT": -0.5}), 1000.0, {}, self.P, self.F, risk2)
        self.assertAlmostEqual(sum(abs(v) for v in plan2.target_notional.values()), 500.0, places=6)
        self.assertAlmostEqual(plan2.gross_scale, 0.25)

    def test_untradable_position_reported(self):
        plan = plan_orders(pd.Series(dtype=float), 1000.0, {"ZZZUSDT": 1.0}, self.P, self.F, RiskCfg())
        self.assertTrue(plan.skipped and "ZZZUSDT" in plan.skipped[0])


class TestPaperBroker(unittest.TestCase):
    def test_accounting(self):
        b = PaperBroker.fresh(1000.0, fee=0.001, slippage=0.0)
        b.execute(Order("AUSDT", "BUY", Decimal("10"), False, "open", 10.0))       # long 10 @10, fee .1
        b.execute(Order("AUSDT", "BUY", Decimal("10"), False, "inc", 20.0))        # avg entry 15, fee .2
        self.assertAlmostEqual(b.book["positions"]["AUSDT"]["entry"], 15.0)
        self.assertAlmostEqual(b.equity({"AUSDT": 20.0}), 1000 - 0.3 + 20 * 5)
        b.execute(Order("AUSDT", "SELL", Decimal("30"), False, "flip", 20.0))      # close 20 (+100), short 10 @20
        self.assertAlmostEqual(b.positions()["AUSDT"], -10.0)
        self.assertAlmostEqual(b.book["positions"]["AUSDT"]["entry"], 20.0)
        self.assertAlmostEqual(b.book["wallet"], 1000 - 0.3 - 0.6 + 100)
        r = b.execute(Order("AUSDT", "SELL", Decimal("5"), True, "bad", 20.0))     # reduce-only same side
        self.assertEqual(r["status"], "REJECTED")
        b.execute(Order("AUSDT", "BUY", Decimal("50"), True, "close", 10.0))       # clamps to 10, +100
        self.assertEqual(b.positions(), {})
        self.assertAlmostEqual(b.book["wallet"], 1000 - 0.3 - 0.6 + 100 + 100 - 0.1)

    def test_funding_and_slippage(self):
        b = PaperBroker.fresh(1000.0, fee=0.0, slippage=0.01)
        f = b.execute(Order("AUSDT", "BUY", Decimal("10"), False, "open", 10.0))
        self.assertAlmostEqual(f["price"], 10.1)
        paid = b.apply_funding({"AUSDT": 0.001}, {"AUSDT": 10.0})
        self.assertAlmostEqual(paid, 0.1)
        self.assertAlmostEqual(b.book["wallet"], 999.9)


class TestRisk(unittest.TestCase):
    def test_drawdown_and_cycle_loss(self):
        r = RiskCfg(max_drawdown=0.3, max_cycle_loss=0.2)
        st = {"peak_equity": 1000.0, "last_equity": 900.0}
        self.assertIsNone(check_risk(st, 800.0, r))
        self.assertIn("drawdown", check_risk(st, 690.0, r))
        st2 = {"peak_equity": 1000.0, "last_equity": 1000.0}
        self.assertIn("cycle loss", check_risk(st2, 790.0, r))
        st3 = {}
        self.assertIsNone(check_risk(st3, 500.0, r)); self.assertEqual(st3["peak_equity"], 500.0)


class Seq:
    """Session returning queued responses / raising queued exceptions."""
    def __init__(self, items):
        self.items, self.calls = list(items), []

    def request(self, method, url, headers=None, timeout=None):
        self.calls.append((method, url, headers))
        x = self.items.pop(0)
        if isinstance(x, Exception):
            raise x
        return x


class TestClient(unittest.TestCase):
    def mk(self, items):
        return BinanceFuturesClient("https://x", "key", "secret", session=Seq(items), sleep=lambda s: None,
                                    clock=lambda: 1_700_000_000.0)

    def test_signature(self):
        c = self.mk([Resp(200, {"ok": 1})])
        c.request("GET", "/fapi/v3/account", {"a": 1}, signed=True)
        url = c.session.calls[0][1]
        q, sig = url.split("?", 1)[1].rsplit("&signature=", 1)
        self.assertEqual(sig, hmac.new(b"secret", q.encode(), hashlib.sha256).hexdigest())
        self.assertIn("timestamp=1700000000000", q)
        self.assertEqual(c.session.calls[0][2]["X-MBX-APIKEY"], "key")

    def test_retries_get_not_post(self):
        c = self.mk([requests.ConnectionError("x"), Resp(502, {}), Resp(429, {}, {"Retry-After": "1"}), Resp(200, {"v": 1})])
        self.assertEqual(c.request("GET", "/fapi/v1/time"), {"v": 1})
        c2 = self.mk([requests.Timeout("t")])
        with self.assertRaises(AmbiguousOrderError):
            c2.new_order(symbol="A", side="BUY", type="MARKET", quantity="1")

    def test_timestamp_resync_and_errors(self):
        c = self.mk([Resp(400, {"code": -1021, "msg": "ts"}), Resp(200, {"serverTime": 1_700_000_005_000}),
                     Resp(200, {"ok": 1})])
        self.assertEqual(c.request("GET", "/fapi/v3/account", signed=True), {"ok": 1})
        self.assertEqual(c.time_offset_ms, 5000)
        c2 = self.mk([Resp(400, {"code": -2019, "msg": "Margin is insufficient."})])
        with self.assertRaises(BinanceAPIError) as e:
            c2.request("POST", "/fapi/v1/order", {}, signed=True)
        self.assertEqual(e.exception.code, -2019)

    def test_parse_filters(self):
        info = {"symbols": [
            {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
             "marginAsset": "USDT", "baseAsset": "BTC", "underlyingType": "COIN",
             "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
                         {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "120"},
                         {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                         {"filterType": "MIN_NOTIONAL", "notional": "100"}]},
            {"symbol": "TSLAUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
             "baseAsset": "TSLA", "filters": []},
            {"symbol": "ETHUSDT_260925", "status": "TRADING", "contractType": "CURRENT_QUARTER", "quoteAsset": "USDT",
             "baseAsset": "ETH", "filters": []},
            {"symbol": "XUSDT", "status": "SETTLING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
             "baseAsset": "X", "filters": []}]}
        f = parse_filters(info, exclude=["TSLA"])
        self.assertEqual(list(f), ["BTCUSDT"])
        self.assertEqual(f["BTCUSDT"].max_qty, Decimal("120"))
        self.assertEqual(f["BTCUSDT"].min_notional, 100.0)
        self.assertEqual(fmt_dec(Decimal("10.000")), "10")
        self.assertEqual(fmt_dec(Decimal("0.00100")), "0.001")


if __name__ == "__main__":
    unittest.main()
