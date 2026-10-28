"""The bot must trade what the research tested: compare bot.strategy (rolling 400-bar window, as
used live) against research/final_v4.py's weights (full history) on several dates."""
import os
import sys
import unittest

import pandas as pd

from bot.config import Config
from bot.data import CsvData
from bot.strategy import compute_targets

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESEARCH = os.path.join(ROOT, "research")
HAS_CACHE = os.path.exists(os.path.join(RESEARCH, "cache", "close_1d.parquet"))


@unittest.skipUnless(HAS_CACHE, "research cache missing: run `cd research && python3 data.py`")
class TestParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, RESEARCH)
        cwd = os.getcwd(); os.chdir(RESEARCH)
        try:
            import engine, strategies as st
            from explore_v4 import load_all, build_v3, I
            from robust_v4 import sleeve
            c, v, r = load_all()
            e = st.eligibility(c, v, 60, 5e6, 30)
            w3 = engine.vol_target(build_v3(c, v, e), r, I, target_vol=0.40)
            wp = sleeve(c, v, r, e, 0.25, 5)[1]
        finally:
            os.chdir(cwd)
        cls.research = (0.7 * w3 + 0.3 * wp).rename(columns=lambda s: s + "USDT")
        cls.cfg = Config()
        cls.data = CsvData(ROOT, cls.cfg.strategy.exclude)

    def test_weights_match(self):
        dates = ["2021-05-20", "2022-06-15", "2023-03-15", "2024-03-05", "2024-11-20", "2025-08-01", "2026-06-06"]
        for d in dates:
            day = pd.Timestamp(d, tz="UTC")
            bars = self.data.daily_bars(self.data.universe(), self.cfg.strategy.history_bars,
                                        day + pd.Timedelta(days=1, minutes=5))
            bot_w = compute_targets(bars.close, bars.volume, self.cfg.strategy).weights
            ref = self.research.loc[day]
            ref = ref[ref.abs() > 1e-9]
            both = sorted(set(bot_w.index) | set(ref.index))
            diff = (bot_w.reindex(both).fillna(0) - ref.reindex(both).fillna(0)).abs().max()
            with self.subTest(day=d):
                self.assertGreater(len(ref), 0)
                self.assertLess(diff, 1e-5, f"{d}: max weight diff {diff:.5f}\nbot:\n{bot_w}\nresearch:\n{ref}")


if __name__ == "__main__":
    unittest.main()
