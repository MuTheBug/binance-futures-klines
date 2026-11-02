import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bot.binance import BinanceFuturesClient
from bot.state import Notifier, StateStore
from gridbot.cli import build
from gridbot.config import Config

from gridbot.sim_exchange import SimExchange
from gridbot.tests.helpers import FakeBinanceHTTP, filters_for

H = 3_600_000


def kl(closes, start_ms, step_ms):
    return [[start_ms + i * step_ms, str(c), str(c * 1.002), str(c * 0.998), str(c), "1000", start_ms + (i + 1) * step_ms - 1]
            for i, c in enumerate(closes)]


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.sim = SimExchange(filters_for(["DOGEUSDT"], step="1", tick="0.00001"), 100.0, leverage=5, slippage=0.0)
        self.sim.set_price("DOGEUSDT", 0.2)
        rng = np.random.default_rng(1)
        hourly = list(0.2 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
        daily = list(0.2 * np.exp(np.cumsum(rng.normal(0, 0.03, 200))))
        self.http = FakeBinanceHTTP(self.sim, klines={("DOGEUSDT", "1h"): kl(hourly, 0, H),
                                                      ("DOGEUSDT", "1d"): kl(daily, 0, 24 * H)})
        self.now = 10 * 24 * H

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def make(self, **risk):
        cfg = Config(mode="live", state_dir=self.dir, symbols=["DOGEUSDT"], api_key="k", api_secret="s")
        cfg.grid.levels, cfg.grid.regime_filter, cfg.grid.exit_rule, cfg.grid.spacing_k = 5, False, "reset", 3.0
        for k, v in risk.items():
            setattr(cfg.risk, k, v)
        client = BinanceFuturesClient("https://fake", "k", "s", session=self.http, sleep=lambda s: None,
                                      clock=lambda: 0.0)
        store = StateStore(self.dir)
        state = store.load()
        bot, *_ = build(cfg, state, store, client=client)
        bot.clock = lambda: self.now
        bot.notifier = Notifier()
        return bot, store

    def move(self, px):
        self.sim.set_price("DOGEUSDT", px)


class TestLiveHttpPath(Base):
    def test_full_cycle_over_http(self):
        bot, store = self.make()
        out = bot.cycle()
        self.assertEqual(out["steps"][0].action, "created")
        posts = [p for m, path, p in self.http.calls if m == "POST" and path == "/fapi/v1/order"]
        self.assertTrue(posts and all(p["type"] == "LIMIT" and p["timeInForce"] == "GTX" for p in posts))
        self.assertTrue(all("e" not in p["quantity"].lower() and p["newClientOrderId"].startswith("g") for p in posts))
        self.assertEqual(len(self.sim.open_orders("DOGEUSDT")), 10)
        g = bot.engine.grid("DOGEUSDT")
        self.move(g.level(-2) * 0.999)                     # two buys fill on the exchange
        self.now += 60_000
        bot.cycle()
        self.assertAlmostEqual(self.sim.pos["DOGEUSDT"], float(g.inv(-2)))
        cids = {o["clientOrderId"] for o in self.sim.open_orders("DOGEUSDT")}
        self.assertIn(g.cid("S", -1), cids)               # opposite orders re-placed via HTTP
        self.assertEqual(len(cids), 10)
        # restart: a fresh bot from disk re-places nothing
        bot2, _ = self.make()
        n_calls = len(self.http.calls)
        out2 = bot2.cycle()
        new_posts = [c for c in self.http.calls[n_calls:] if c[0] == "POST" and c[1] == "/fapi/v1/order"]
        self.assertEqual(new_posts, [])
        self.assertEqual(out2["steps"][0].action, "ok")


class TestRiskLayer(Base):
    def test_drawdown_halt_flattens_and_persists(self):
        bot, store = self.make(max_drawdown=0.05, exposure=3.0)
        bot.cycle()
        g = bot.engine.grid("DOGEUSDT")
        self.move(g.level(-5) * 0.97)                      # fills every buy, then keeps falling
        self.now += 60_000
        bot.cycle()
        self.move(g.level(-5) * 0.80)
        self.now += 60_000
        out = bot.cycle()
        self.assertEqual(out["status"], "halted")
        self.assertEqual(self.sim.pos["DOGEUSDT"], 0.0)
        self.assertEqual(self.sim.open_orders("DOGEUSDT"), [])
        bot2, _ = self.make()
        self.assertEqual(bot2.cycle()["status"], "halted")

    def test_stop_file_pauses(self):
        bot, store = self.make()
        Path(self.dir, "STOP").touch()
        self.assertEqual(bot.cycle()["status"], "paused")
        self.assertEqual(self.sim.open_orders("DOGEUSDT"), [])

    def test_whipsaw_guard(self):
        bot, store = self.make(max_resets_per_day=2, max_daily_loss=0.95, max_drawdown=0.95)
        bot.cfg.grid.cooldown_minutes = 60
        bot.cycle()
        px = 0.2
        for i in range(4):                                 # repeated range exits within one day
            px *= 1.3 if i % 2 == 0 else 0.75
            self.move(px)
            self.now += 60_000
            bot.cycle()
        self.assertGreater(bot.state["cooldown_until"].get("DOGEUSDT", 0), self.now)
        self.assertEqual(self.sim.pos["DOGEUSDT"], 0.0)


class TestPaperMode(unittest.TestCase):
    def test_paper_state_survives_restart(self):
        d = tempfile.mkdtemp()
        try:
            sim_public = SimExchange(filters_for(["DOGEUSDT"], step="1", tick="0.00001"), 0.0)
            sim_public.set_price("DOGEUSDT", 0.2)
            rng = np.random.default_rng(2)
            minutes = kl(list(0.2 * np.exp(np.cumsum(rng.normal(0, 0.002, 5)))), 9 * 24 * H, 60_000)
            http = FakeBinanceHTTP(sim_public, klines={
                ("DOGEUSDT", "1h"): kl(list(0.2 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))), 0, H),
                ("DOGEUSDT", "1d"): kl(list(0.2 * np.exp(np.cumsum(rng.normal(0, 0.03, 200)))), 0, 24 * H),
                ("DOGEUSDT", "1m"): minutes})
            cfg = Config(mode="paper", state_dir=d, symbols=["DOGEUSDT"])
            cfg.grid.levels, cfg.grid.regime_filter = 5, False

            def mk():
                client = BinanceFuturesClient("https://fake", session=http, sleep=lambda s: None)
                store = StateStore(d)
                bot, *_ = build(cfg, store.load(), store, client=client)
                bot.clock = lambda: 9 * 24 * H + 600_000
                return bot
            b1 = mk()
            b1.cycle()
            eq1, orders1 = b1.equity(), len(b1.ex.open_orders("DOGEUSDT"))
            self.assertGreater(orders1, 0)
            b2 = mk()                                       # new process: paper account restored
            self.assertAlmostEqual(b2.equity(), eq1)
            self.assertEqual(len(b2.ex.open_orders("DOGEUSDT")), orders1)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


class TestRotation(unittest.TestCase):
    def test_max_active_slots_rotate(self):
        d = tempfile.mkdtemp()
        try:
            syms = ["AUSDT", "BUSDT", "CUSDT"]
            sim = SimExchange(filters_for(syms, step="0.1", tick="0.0001"), 100.0, leverage=5, slippage=0.0)
            rng = np.random.default_rng(5)
            kls = {}
            for s in syms:
                sim.set_price(s, 1.0)
                kls[(s, "1h")] = kl(list(np.exp(np.cumsum(rng.normal(0, 0.01, 300)))), 0, H)
                kls[(s, "1d")] = kl(list(np.exp(np.cumsum(rng.normal(0, 0.03, 200)))), 0, 24 * H)
            http = FakeBinanceHTTP(sim, klines=kls)
            cfg = Config(mode="live", state_dir=d, symbols=syms, max_active=1, api_key="k", api_secret="s")
            cfg.grid.levels, cfg.grid.exit_rule, cfg.grid.cooldown_minutes = 4, "stop", 600
            cfg.grid.regime_filter, cfg.grid.spacing_k = False, 3.0
            cfg.risk.max_daily_loss, cfg.risk.max_drawdown = 0.9, 0.9
            client = BinanceFuturesClient("https://fake", "k", "s", session=http, sleep=lambda s: None, clock=lambda: 0.0)
            store = StateStore(d)
            bot, *_ = build(cfg, store.load(), store, client=client)
            now = [10 * 24 * H]
            bot.clock = lambda: now[0]
            bot.notifier = Notifier()
            bot.cycle()
            self.assertEqual(list(bot.state["grids"]), ["AUSDT"])          # one slot, first candidate
            g = bot.engine.grid("AUSDT")
            self.assertAlmostEqual(g.q_quote * g.n, 100.0, delta=12.0)      # sized from the whole slot
            sim.set_price("AUSDT", g.level(-g.n) * 0.85)                    # A breaks down -> stopped
            now[0] += 60_000
            bot.cycle()
            self.assertNotIn("AUSDT", bot.state["grids"])
            now[0] += 60_000
            bot.cycle()
            self.assertEqual(list(bot.state["grids"]), ["BUSDT"])           # slot passed on to B
            self.assertEqual(sim.pos["CUSDT"], 0.0)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestCliPlan(Base):
    def test_plan_prints_grid_without_trading(self):
        import contextlib
        import io
        from gridbot.cli import cmd_plan
        bot, store = self.make()
        client = bot.feed.client
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd_plan(bot.cfg, bot, client, bot.filters, bot.strategy)
        out = buf.getvalue()
        self.assertIn("DOGEUSDT", out)
        self.assertIn("USDT per level", out)
        self.assertEqual(self.sim.open_orders("DOGEUSDT"), [])          # plan never trades
        self.assertEqual(self.sim.pos["DOGEUSDT"], 0.0)
