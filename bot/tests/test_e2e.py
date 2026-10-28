"""End-to-end: full bot cycles against the fake Binance exchange (real client, signing, data
parsing, order planning, brokers, state, risk), in paper and live mode."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from bot.binance import BinanceFuturesClient
from bot.brokers import LiveBroker, PaperBroker
from bot.config import Config
from bot.data import BinanceData, CsvData
from bot.runner import Bot, StaleDataError
from bot.state import InstanceLock, StateStore
from bot.tests.fake_binance import FakeBinance

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DAY1 = pd.Timestamp("2024-03-06 00:05", tz="UTC")      # pump sleeve active on 2024-03-05
_CSV = None


def csv():
    global _CSV
    if _CSV is None:
        _CSV = CsvData(ROOT, Config().strategy.exclude)
    return _CSV


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.fake = FakeBinance(csv(), DAY1)
        self.client = BinanceFuturesClient("https://fake", "k", "s", session=self.fake, sleep=lambda s: None,
                                           clock=lambda: self.fake.now.timestamp())
        self.data = BinanceData(self.client, Config().strategy.exclude, min_quote_volume_24h=0)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def cfg(self, mode="paper", **risk):
        c = Config(mode=mode, state_dir=self.dir)
        c.risk.leverage = 1.0
        for k, v in risk.items():
            setattr(c.risk, k, v)
        return c


class TestPaperE2E(Base):
    def bot(self, cfg, state=None):
        store = StateStore(self.dir)
        state = store.load() if state is None else state
        broker = PaperBroker(state.get("paper") or {"wallet": 10_000.0, "positions": {}}, 0.0005, 0.0002)
        return Bot(cfg, self.data, broker, store, state, clock=lambda: self.fake.now)

    def test_cycle_idempotent_and_next_day(self):
        cfg = self.cfg()
        bot = self.bot(cfg)
        r = bot.run_cycle()
        self.assertEqual(r.status, "done")
        self.assertEqual(r.day, "2024-03-05")
        pos = bot.broker.positions()
        marks = self.fake.marks()
        # executed notionals track the target weights (equity 10k, leverage 1)
        for s, w in r.targets.items():
            if abs(w) * 10_000 > 50:
                self.assertAlmostEqual(pos.get(s, 0) * marks[s] / 10_000, w, delta=0.011)
        self.assertIn("1000PEPEUSDT", pos)                 # pump sleeve long
        self.assertLess(pos["BTCUSDT"] * marks["BTCUSDT"], r.targets.get("BTCUSDT", 0) * 10_000 + 200)
        self.assertEqual(bot.run_cycle().status, "already-done")
        # restart from disk, next day: funding applied, trades again
        self.fake.now = DAY1 + pd.Timedelta(days=1)
        bot2 = self.bot(cfg)
        self.assertEqual(bot2.broker.positions(), pos)
        r2 = bot2.run_cycle()
        self.assertEqual(r2.status, "done")
        self.assertNotEqual(bot2.broker.book["funding"], 0.0)
        lines = Path(self.dir, "equity.csv").read_text().splitlines()
        self.assertEqual(len(lines), 3)
        events = [json.loads(x)["event"] for x in Path(self.dir, "journal.jsonl").read_text().splitlines()]
        self.assertIn("paper_funding", events)

    def test_stale_data_and_stop_file(self):
        cfg = self.cfg()
        bot = self.bot(cfg)
        # the exchange's latest bar is 2024-03-05 but the clock says 03-07 closed -> refuse to trade
        with self.assertRaises(StaleDataError):
            bot.run_cycle(now=pd.Timestamp("2024-03-08 00:05", tz="UTC"))
        Path(self.dir, "STOP").touch()
        self.assertEqual(bot.run_cycle().status, "paused")

    def test_drawdown_halts_and_flattens(self):
        cfg = self.cfg(max_drawdown=0.05)
        bot = self.bot(cfg)
        bot.run_cycle()
        pos = bot.broker.positions()
        self.fake.now = DAY1 + pd.Timedelta(days=1)
        # crash every long 60% and squeeze every short 60%
        m = self.fake.marks()
        self.fake.marks_override = {s: m[s] * (0.4 if q > 0 else 1.6) for s, q in pos.items()}
        r = bot.run_cycle()
        self.assertEqual(r.status, "halted")
        self.assertEqual(bot.broker.positions(), {})
        self.assertTrue(StateStore(self.dir).load()["halted"])
        self.assertEqual(bot.run_cycle().status, "halted")

    def test_dry_run_trades_nothing(self):
        bot = self.bot(self.cfg())
        r = bot.run_cycle(dry_run=True)
        self.assertEqual(r.status, "dry-run")
        self.assertTrue(r.orders)
        self.assertEqual(bot.broker.positions(), {})


class TestLiveE2E(Base):
    def bot(self, cfg, **kw):
        store = StateStore(self.dir)
        state = store.load()
        broker = LiveBroker(self.client, self.data.filters(), "CROSSED", 5, state.setdefault("live_configured", {}),
                            sleep=lambda s: None)
        broker.prepare()
        return Bot(cfg, self.data, broker, store, state, clock=lambda: self.fake.now, **kw)

    def test_live_cycle(self):
        cfg = self.cfg("live", stop_loss_pct=0.3)
        bot = self.bot(cfg)
        r = bot.run_cycle()
        self.assertEqual(r.status, "done")
        self.assertTrue(all(f["status"] == "FILLED" for f in r.fills), r.fills)
        self.assertEqual(set(self.fake.positions), set(bot.state["owned"]))
        self.assertTrue(all(v == 5 for v in self.fake.leverage.values()))
        self.assertEqual(len(self.fake.open_stops), len(self.fake.positions))
        for o in self.fake.open_stops.values():
            self.assertEqual(o["closePosition"], "true")
        posts = [p for m, path, p in self.fake.requests if m == "POST" and path == "/fapi/v1/order" and p["type"] == "MARKET"]
        reduce_idx = [i for i, p in enumerate(posts) if p.get("reduceOnly") == "true"]
        open_idx = [i for i, p in enumerate(posts) if p.get("reduceOnly") != "true"]
        self.assertTrue(not reduce_idx or max(reduce_idx) < min(open_idx))
        # next day: stops replaced, not duplicated
        self.fake.now = DAY1 + pd.Timedelta(days=1)
        bot2 = self.bot(cfg)
        r2 = bot2.run_cycle()
        self.assertEqual(r2.status, "done")
        self.assertEqual(len(self.fake.open_stops), len(self.fake.positions))
        self.assertAlmostEqual(bot2.state["last_equity"], self.fake.equity(), places=4)

    def test_ambiguous_order_is_not_duplicated(self):
        bot = self.bot(self.cfg("live"))
        self.fake.fail_next_post = 1           # first order fills but the response is a 503
        r = bot.run_cycle()
        self.assertEqual(r.fills[0]["status"], "FILLED")
        self.assertEqual(len(self.fake.orders), len(r.fills))

    def test_refuses_foreign_positions_and_hedge_mode(self):
        self.fake._apply("ETHUSDT", 1.0, self.fake.marks()["ETHUSDT"])
        bot = self.bot(self.cfg("live"))
        with self.assertRaisesRegex(RuntimeError, "did not open"):
            bot.run_cycle()
        self.fake.dual = True
        with self.assertRaisesRegex(RuntimeError, "Hedge Mode"):
            self.bot(self.cfg("live"))

    def test_mode_mismatch_and_lock(self):
        StateStore(self.dir).save({"mode": "paper"})
        with self.assertRaisesRegex(RuntimeError, "belongs to mode"):
            self.bot(self.cfg("live"))
        with InstanceLock(self.dir):
            with self.assertRaisesRegex(RuntimeError, "another bot instance"):
                with InstanceLock(self.dir):
                    pass


class TestScheduler(Base):
    def test_run_forever_trades_once_per_day(self):
        cfg = self.cfg()
        cfg.schedule.run_minute_utc = 5
        store = StateStore(self.dir)
        broker = PaperBroker.fresh(10_000.0, 0.0005, 0.0002)
        times = iter([pd.Timestamp("2024-03-06 00:01", tz="UTC"), DAY1, pd.Timestamp("2024-03-06 07:00", tz="UTC"),
                      pd.Timestamp("2024-03-07 00:06", tz="UTC")])
        bot = Bot(cfg, self.data, broker, store, {}, sleep=lambda s: None)

        def clock():
            try:
                t = next(times)
            except StopIteration:
                bot._stop = True
                return pd.Timestamp("2024-03-07 01:00", tz="UTC")
            self.fake.now = t
            return t
        bot.clock = clock
        bot.run_forever()
        days = [l.split(",")[0] for l in Path(self.dir, "equity.csv").read_text().splitlines()[1:]]
        self.assertEqual(days, ["2024-03-05", "2024-03-06"])


if __name__ == "__main__":
    unittest.main()
