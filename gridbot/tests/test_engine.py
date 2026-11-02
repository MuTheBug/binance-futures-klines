import os
import sys
import unittest

import numpy as np
import pandas as pd

from gridbot.engine import GridEngine, GridPlan
from gridbot.replay import replay
from gridbot.sim_exchange import SimExchange
from gridbot.tests.helpers import filters_for, random_walk_bars

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "research"))
from gridsim import GridParams, simulate  # noqa: E402


def research_equity(bars, **kw):
    o, h, l, c = (bars[[k]].rename(columns={k: "X"}) for k in ("open", "high", "low", "close"))
    return simulate(o, h, l, c, GridParams(**kw))["equity"]["X"]


class TestParityWithResearch(unittest.TestCase):
    """The production engine + exchange simulator must trade exactly what the research simulator
    modelled (up to lot-size rounding)."""

    def test_random_walks(self):
        cases = [("neutral", "reset", 0), ("long", "reset", 0), ("short", "hold", 0), ("neutral", "stop", 24),
                 ("long", "hold", 0)]
        for seed, (bias, rule, cool) in enumerate(cases):
            bars = random_walk_bars(1200, seed=seed + 10, vol=0.012)
            ref = research_equity(bars, g=0.015, n=6, bias=bias, on_exit=rule, exit_buffer=1, cooldown=cool,
                                  compound=True)
            eq, ex = replay({"XUSDT": bars}, lambda s, t, b=bias: GridPlan(True, g=0.015, bias=b), n=6,
                            exit_rule=rule, exit_buffer=1, cooldown_s=cool * 3600)
            with self.subTest(bias=bias, rule=rule):
                self.assertLess((eq.to_numpy() - ref.to_numpy()).__abs__().max(), 2e-4)
                self.assertGreater(len(ex.trades), 50)

    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, "research", "cache", "close_1h.parquet")),
                         "research cache missing")
    def test_real_data_wick(self):
        import data
        sym = "SOL"
        o, h, l, c = (data.load(k, "1h")[[sym]].astype("float64").loc["2024-01-01":"2024-02-15"]
                      for k in ("open", "high", "low", "close"))
        bars = pd.DataFrame({"open": o[sym], "high": h[sym], "low": l[sym], "close": c[sym]})
        ref = research_equity(bars, g=0.01, n=8, bias="short", on_exit="hold", compound=True)
        eq, _ = replay({"XUSDT": bars}, lambda s, t: GridPlan(True, g=0.01, bias="short"), n=8, exit_rule="hold")
        self.assertLess(np.abs(eq.to_numpy() - ref.to_numpy()).max(), 2e-4)   # includes the Jan-3 -18% wick


class TestEngineRobustness(unittest.TestCase):
    def setUp(self):
        self.f = filters_for(["XUSDT"])
        self.sim = SimExchange(self.f, 100.0, leverage=5, slippage=0.0)
        self.sim.set_price("XUSDT", 100.0)
        self.state = {}
        self.eng = GridEngine(self.sim, self.f, self.state, n=5, exposure=1.0, exit_rule="reset", exit_buffer=1)
        self.plan = lambda: GridPlan(True, g=0.02, bias="neutral")

    def step(self, px, now=0, eng=None, check_exit=True):
        self.sim.set_price("XUSDT", px)
        return (eng or self.eng).step("XUSDT", px, self.plan, 100.0, now, f"t{now}", check_exit=check_exit)

    def test_create_and_restart_is_idempotent(self):
        r = self.step(100.0)
        self.assertEqual(r.action, "created")
        self.assertEqual(r.placed, 10)
        eng2 = GridEngine(self.sim, self.f, self.state, n=5, exposure=1.0)       # restart from saved state
        r2 = self.step(100.0, eng=eng2)
        self.assertEqual((r2.placed, r2.cancelled, r2.market_trades), (0, 0, []))

    def test_fills_flip_orders(self):
        self.step(100.0)
        self.step(97.5)                                    # crosses 98.0 -> buy filled
        g = self.eng.grid("XUSDT")
        self.assertAlmostEqual(self.sim.pos["XUSDT"], float(g.unit(-1)))
        cids = {o["clientOrderId"] for o in self.sim.open_orders("XUSDT")}
        self.assertIn(g.cid("S", 0), cids)                  # sell placed one level up
        self.assertNotIn(g.cid("B", -1), cids)
        self.step(100.5)                                   # back up: sold again -> flat, +1 round trip
        self.assertAlmostEqual(self.sim.pos["XUSDT"], 0.0)

    def test_catch_up_after_downtime(self):
        self.step(100.0)
        self.sim.cancel_all_open_orders("XUSDT")           # outage: our orders vanished
        self.sim.set_price("XUSDT", 93.0)                  # price fell through 3 levels unseen
        r = self.eng.step("XUSDT", 93.0, self.plan, 100.0, 1, "t1")
        g = self.eng.grid("XUSDT")
        self.assertTrue(any("catch-up" in t["reason"] for t in r.market_trades))
        self.assertAlmostEqual(self.sim.pos["XUSDT"], float(g.inv(-3)), places=6)
        self.assertEqual(len(self.sim.open_orders("XUSDT")), 10)   # rebuilt: buys -5,-4; sells -2..5

    def test_post_only_rejection_is_not_an_error(self):
        self.step(100.0)
        self.sim.cancel_all_open_orders("XUSDT")
        self.sim.set_price("XUSDT", 97.0)                  # engine is told a stale price of 100
        r = self.eng.step("XUSDT", 100.0, self.plan, 100.0, 1, "t1", check_exit=False)
        self.assertEqual(r.errors, [])                     # the -5022 rejections are expected, skipped
        r2 = self.step(97.0, now=2)                        # next step with the true price catches up
        self.assertTrue(r2.market_trades)

    def test_stop_and_cooldown(self):
        eng = GridEngine(self.sim, self.f, self.state, n=3, exposure=1.0, exit_rule="stop", exit_buffer=1,
                         cooldown_s=3600)
        self.step(100.0, eng=eng)
        r = self.step(90.0, now=1000, eng=eng)            # below 100*1.02^-4
        self.assertEqual(r.action, "stopped")
        self.assertAlmostEqual(self.sim.pos["XUSDT"], 0.0)
        self.assertEqual(self.sim.open_orders("XUSDT"), [])
        self.assertEqual(self.step(90.0, now=2000, eng=eng).action, "cooldown")
        self.assertEqual(self.step(90.0, now=3_700_000, eng=eng).action, "created")

    def test_small_capital_uses_fewer_levels(self):
        eng = GridEngine(self.sim, self.f, self.state, n=10, exposure=1.0)
        g = eng.new_grid("XUSDT", 100.0, GridPlan(True, g=0.02, bias="neutral"), 30.0, 0)
        self.assertLess(g.n, 10)
        self.assertGreaterEqual(g.q_quote, 5.0 * 1.02)
        self.assertIsNone(eng.new_grid("XUSDT", 100.0, GridPlan(True, g=0.02, bias="neutral"), 3.0, 0))
        coarse = filters_for(["XUSDT"], step="1")                 # 1 unit = 100 USDT per order
        eng_c = GridEngine(self.sim, coarse, {}, n=10, exposure=1.0)
        self.assertIsNone(eng_c.new_grid("XUSDT", 100.0, GridPlan(True, g=0.02, bias="neutral"), 60.0, 0))
        g = eng_c.new_grid("XUSDT", 100.0, GridPlan(True, g=0.02, bias="neutral"), 400.0, 0)
        self.assertLessEqual(g.max_notional(), 400.0 * 1.25)      # rounding never inflates exposure


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.path.exists(os.path.join(ROOT, "research", "cache", "close_1h.parquet")),
                     "research cache missing")
class TestLeadingDesignParity(unittest.TestCase):
    """The design the research ranked first (vol-scaled neutral grid, daily trend filter, stop with a
    24h cool-down) must behave identically in the bot, including the filter and the cool-down."""

    def test_regime_filtered_stop_grid(self):
        import data
        from grid_study import daily_trend_on_hours
        from gridbot.config import Config
        from gridbot.replay_cli import make_plan_fn
        sym = "DOGE"
        frames = {k: data.load(k, "1h")[[sym]].astype("float64").loc["2023-09-01":"2024-06-30"]
                  for k in ("open", "high", "low", "close")}
        bars = pd.DataFrame({k: v[sym] for k, v in frames.items()})
        start = pd.Timestamp("2024-01-01", tz="UTC")
        trend = daily_trend_on_hours({"close": frames["close"]})[sym]
        sigma = np.log(frames["close"][sym]).diff().rolling(168, min_periods=84).std()
        allow = (trend.abs() < 0.3) & (bars.index >= start)
        gp = GridParams(g=0.01, n=10, bias="neutral", on_exit="stop", exit_buffer=1, cooldown=24, compound=True)
        o, h, l, c = (frames[k].rename(columns={sym: "X"}) for k in ("open", "high", "low", "close"))
        ref = simulate(o, h, l, c, gp, g_frame=(sigma * 3.0).to_frame("X"), allow=allow.to_frame("X"))["equity"]["X"]

        cfg = Config()
        cfg.grid.spacing_k, cfg.grid.levels, cfg.grid.regime_filter = 3.0, 10, True
        cfg.grid.spacing_min, cfg.grid.spacing_max = 1e-6, 1.0
        plan = make_plan_fn(cfg, bars)
        eq, ex = replay({"XUSDT": bars}, lambda s, t: plan(s, t) if t >= start else GridPlan(False),
                        n=10, exit_rule="stop", exit_buffer=1, cooldown_s=24 * 3600,
                        regime_fn=lambda s, t: bool(allow.get(t, False)))
        diff = np.abs(eq.to_numpy() - ref.to_numpy())
        self.assertGreater(len(ex.trades), 100)
        self.assertLess(diff.max(), 1e-3, f"max diff {diff.max():.2e} at {eq.index[diff.argmax()]}")
