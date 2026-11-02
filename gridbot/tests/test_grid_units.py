import os
import sys
import tempfile
import unittest
from decimal import Decimal

import numpy as np

from bot.binance import BinanceAPIError
from gridbot.config import load_config
from gridbot.grid import Grid, q_for_exposure
from gridbot.sim_exchange import SimExchange
from gridbot.strategy import hourly_sigma, trend_strength
from gridbot.tests.helpers import filters_for

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mkgrid(bias="neutral", n=5, g=0.01, q=10.0, step="0.001", min_notional=5.0):
    return Grid("XUSDT", "abc123", 100.0, g, n, q, bias, step, "0.01", min_notional)


class TestGrid(unittest.TestCase):
    def test_inventory_shape(self):
        for bias in ("long", "neutral", "short"):
            g = mkgrid(bias)
            inv = [float(g.inv(r)) for r in range(-g.n, g.n + 1)]
            self.assertTrue(all(a > b for a, b in zip(inv, inv[1:])), bias)   # falls as price rises
        self.assertEqual(float(mkgrid("long").inv(5)), 0.0)
        self.assertEqual(float(mkgrid("neutral").inv(0)), 0.0)
        self.assertEqual(float(mkgrid("short").inv(-5)), 0.0)

    def test_units_respect_exchange_minimum(self):
        g = mkgrid(q=3.0, min_notional=5.0)                   # 3 USDT asked, exchange needs 5
        for k in range(-g.n, g.n):
            self.assertGreaterEqual(float(g.unit(k)) * g.level(k), 5.0)
            self.assertEqual(g.unit(k) % Decimal("0.001"), 0)

    def test_position_mapping_and_hysteresis(self):
        g = mkgrid()
        for r in range(-5, 6):
            self.assertEqual(g.r_from_position(float(g.inv(r)))[0], r)
        self.assertEqual(g.hysteresis(0, 100.0 * 1.01 ** -2.5), -2)       # fell through -1, -2
        self.assertEqual(g.hysteresis(0, 100.0 * 1.01 ** 1.2), 1)
        self.assertEqual(g.hysteresis(-2, 100.0 * 1.01 ** -1.5), -2)      # inside the hysteresis band
        self.assertEqual(g.hysteresis(0, 100.0 * 1.01 ** -1.1, margin=0.3), 0)   # just touched: wait
        self.assertEqual(g.hysteresis(0, 50.0), -5)                       # clamped at the range edge
        self.assertEqual(g.out_of_range(100 * 1.01 ** -6.5, 1), -1)
        self.assertEqual(g.out_of_range(100 * 1.01 ** 5.5, 1), 0)

    def test_ladder(self):
        g = mkgrid()
        lad = g.ladder(0, 10)
        buys = sorted(o["k"] for o in lad if o["side"] == "BUY")
        sells = sorted(o["k"] for o in lad if o["side"] == "SELL")
        self.assertEqual(buys, [-5, -4, -3, -2, -1])
        self.assertEqual(sells, [1, 2, 3, 4, 5])
        s1 = next(o for o in lad if o["k"] == 1)
        self.assertEqual(s1["qty"], g.unit(0))                            # sells what level 0 bought
        self.assertTrue(all(o["price"] % Decimal("0.01") == 0 for o in lad))
        self.assertEqual(len({o["cid"] for o in lad}), len(lad))
        self.assertTrue(all(len(o["cid"]) <= 36 and g.owns(o["cid"]) for o in lad))

    def test_q_for_exposure_matches_research(self):
        sys.path.insert(0, os.path.join(ROOT, "research"))
        from gridsim import _q_for
        for bias in ("long", "neutral", "short"):
            self.assertAlmostEqual(q_for_exposure(bias, 100.0, 0.02, 7, 1.5),
                                   float(_q_for(bias, np.array([100.0]), 1 / 1.02, 7, 1.5)[0]))


class TestStrategyFormulas(unittest.TestCase):
    def test_sigma_and_trend(self):
        rng = np.random.default_rng(0)
        c = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
        r = np.diff(np.log(c[-169:]))
        self.assertAlmostEqual(hourly_sigma(c), float(np.std(r, ddof=1)))
        up = list(100 * 1.01 ** np.arange(200) * np.exp(rng.normal(0, 0.005, 200)))
        self.assertGreater(trend_strength(up), 0.8)
        self.assertLess(trend_strength(up[::-1]), -0.8)
        self.assertTrue(np.isnan(trend_strength(up[:50])))


class TestSimExchange(unittest.TestCase):
    def setUp(self):
        self.sim = SimExchange(filters_for(["XUSDT"]), 100.0, leverage=2, maker_fee=0.001, taker_fee=0.002,
                               slippage=0.0)
        self.sim.set_price("XUSDT", 100.0)

    def test_post_only_and_limit_fills(self):
        with self.assertRaises(BinanceAPIError) as e:
            self.sim.new_order(symbol="XUSDT", side="BUY", type="LIMIT", timeInForce="GTX", quantity="0.1",
                               price="100.0", newClientOrderId="a")
        self.assertEqual(e.exception.code, -5022)
        self.sim.new_order(symbol="XUSDT", side="BUY", type="LIMIT", timeInForce="GTX", quantity="0.1",
                           price="99.0", newClientOrderId="b")
        self.sim.set_price("XUSDT", 99.5)
        self.assertEqual(self.sim.pos["XUSDT"], 0.0)
        self.sim.set_price("XUSDT", 98.0)                     # trades through 99 -> filled AT 99
        self.assertAlmostEqual(self.sim.pos["XUSDT"], 0.1)
        self.assertAlmostEqual(self.sim.trades[-1]["price"], 99.0)
        self.assertAlmostEqual(self.sim.wallet, 100.0 - 0.1 * 99 * 0.001)
        self.assertAlmostEqual(self.sim.equity(), 100.0 - 0.0099 + 0.1 * (98 - 99))

    def test_filters_margin_reduce_only(self):
        bad = dict(symbol="XUSDT", side="BUY", type="LIMIT", timeInForce="GTX", price="90.0")
        with self.assertRaises(BinanceAPIError):
            self.sim.new_order(quantity="0.0005", **bad)      # off-step
        with self.assertRaises(BinanceAPIError):
            self.sim.new_order(quantity="0.01", **bad)        # 0.9 USDT < 5 min notional
        with self.assertRaises(BinanceAPIError) as e:
            self.sim.new_order(quantity="3", **bad)           # 270 USDT > 100 * 2x leverage
        self.assertEqual(e.exception.code, -2019)
        with self.assertRaises(BinanceAPIError):
            self.sim.new_order(symbol="XUSDT", side="SELL", type="MARKET", quantity="0.1", reduceOnly="true")

    def test_liquidation_and_persistence(self):
        self.sim.new_order(symbol="XUSDT", side="BUY", type="MARKET", quantity="1.9")
        self.sim.set_price("XUSDT", 47.5)                     # equity 0.63 < 1% maintenance -> liquidated
        self.assertTrue(self.sim.liquidated)
        self.assertEqual(self.sim.pos["XUSDT"], 0.0)
        d = self.sim.to_dict()
        other = SimExchange(filters_for(["XUSDT"]), 0.0)
        other.load_dict(d)
        self.assertAlmostEqual(other.wallet, self.sim.wallet)
        self.assertTrue(other.liquidated)


class TestConfig(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(load_config(None, env={}).mode, "paper")
        bad = ['mode = "live"\n', 'symbols = ["BTC"]\n', '[risk]\nexposure = 9.0\n', '[grid]\nexit_rule = "pray"\n',
               'api_key = "x"\n', 'symbols = ["AUSDT", "BUSDT"]\nallocation = [0.7, 0.7]\n', '[grid]\nlevelz = 3\n']
        for text in bad:
            with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
                f.write(text)
            with self.assertRaises(ValueError, msg=text):
                load_config(f.name, env={})
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()


class TestReplayCliFilter(unittest.TestCase):
    def test_regime_fn_matches_plan_and_turns_off_in_trends(self):
        import pandas as pd
        from gridbot.config import Config
        from gridbot.replay_cli import make_plan_fn, make_regime_fn
        idx = pd.date_range("2024-01-01", periods=24 * 200, freq="h", tz="UTC")
        rng = np.random.default_rng(4)
        flat = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
        trend = flat * np.exp(np.linspace(0, 2.5, len(idx)))           # strong steady uptrend
        cfg = Config()
        for series, expect in ((trend, False),):
            bars = pd.DataFrame({"open": series, "high": series, "low": series, "close": series}, index=idx)
            regime, plan = make_regime_fn(cfg, bars), make_plan_fn(cfg, bars)
            t = idx[-1]
            self.assertEqual(regime("X", t), expect)
            self.assertEqual(plan("X", t).allowed, expect)              # creation and running agree
