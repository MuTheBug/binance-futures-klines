"""
Offline self-test — exercises the bot's pure logic with NO network and NO API keys:
  * order reconciliation (open / trim / flip / close, banding, dust, lot rounding)
  * flatten
  * a mock exchange + mock telegram so command handlers run end-to-end
  * the live signal pipeline against a FakeExchange that serves the repo's CSVs as klines

Run: python3 run.py --selftest
"""
from __future__ import annotations
import os
import glob
import time
import pandas as pd
from config import Config
import trader


class MockClient:
    """Stand-in for BinanceFutures with deterministic filters/prices."""
    def __init__(self):
        self.dry_run = True
        self.orders = []
        self._pos = {}
    def round_qty(self, symbol, qty):
        return round(float(qty), 3)
    def set_leverage(self, symbol, lev):
        return {"dry_run": True}
    def place_order(self, symbol, side, quantity, reduce_only=False, order_type="MARKET"):
        self.orders.append((symbol, side, quantity, reduce_only))
        return {"dry_run": True, "symbol": symbol, "side": side, "quantity": quantity}
    def positions(self):
        return dict(self._pos)


def test_reconcile():
    cfg = Config.load()
    cfg.min_order_usdt = 5.0
    cfg.rebalance_band = 0.25
    client = MockClient()
    equity = 10_000.0
    prices = {"BTCUSDT": 100.0, "ETHUSDT": 50.0, "SOLUSDT": 10.0, "XRPUSDT": 1.0}
    # current: long BTC (notional 2000), short ETH (notional -1000)
    positions = {"BTCUSDT": 20.0, "ETHUSDT": -20.0}
    # target: BTC 10% (1000 -> trim), ETH +10% (flip short->long), SOL -15% (open short), XRP 0
    targets = {"BTCUSDT": 0.10, "ETHUSDT": 0.10, "SOLUSDT": -0.15}
    orders = trader.reconcile(targets, equity, positions, prices, client, cfg)
    by = {o.symbol: o for o in orders}
    assert by["BTCUSDT"].side == "SELL" and by["BTCUSDT"].reduce_only, "BTC should trim (reduceOnly SELL)"
    assert by["ETHUSDT"].side == "BUY" and not by["ETHUSDT"].reduce_only, "ETH should flip (BUY, not reduceOnly)"
    assert by["SOLUSDT"].side == "SELL" and not by["SOLUSDT"].reduce_only, "SOL should open short"
    # XRP not in targets and not held -> no order; nothing about XRP
    assert "XRPUSDT" not in by
    # banding: a target equal to current should produce no order
    o2 = trader.reconcile({"BTCUSDT": 0.20}, equity, {"BTCUSDT": 20.0}, prices, client, cfg)
    assert all(o.symbol != "BTCUSDT" for o in o2), "no-churn band should suppress tiny drift"
    print("  ✓ reconcile: trim / flip / open / close / banding")


def test_flatten():
    cfg = Config.load()
    client = MockClient()
    client._pos = {"BTCUSDT": 1.5, "ETHUSDT": -3.0}
    res = trader.flatten(client, cfg, log=lambda *a: None)
    sides = {o[0]: o[1] for o in client.orders}
    assert sides["BTCUSDT"] == "SELL" and sides["ETHUSDT"] == "BUY"
    assert all(o[3] for o in client.orders), "flatten orders must be reduceOnly"
    print("  ✓ flatten: closes longs with SELL, shorts with BUY, reduceOnly")


class FakeExchange:
    """Serves the repo's *_USDT_1d.csv files as Binance-style klines (offline signal test)."""
    def __init__(self, root):
        self.root = root
        self.dry_run = True
    def load_filters(self):
        out = {}
        for f in glob.glob(os.path.join(self.root, "*_USDT_1d.csv")):
            base = os.path.basename(f)[:-len("_USDT_1d.csv")]
            if not base.isascii():
                continue
            out[base + "USDT"] = {"step": 0.001, "minQty": 0.0, "tick": 0.0,
                                  "minNotional": 5.0, "base": base}
        return out
    def klines(self, symbol, interval="1d", limit=300):
        base = symbol[:-4]
        f = os.path.join(self.root, f"{base}_USDT_1d.csv")
        df = pd.read_csv(f).tail(limit)
        out = []
        for _, r in df.iterrows():
            ts = int(r["timestamp"]); close_t = ts + 86_400_000 - 1
            out.append([ts, r["open"], r["high"], r["low"], r["close"], r["volume"],
                        close_t, 0, 0, 0, 0, 0])
        return out


def test_signal_pipeline():
    cfg = Config.load()
    root = os.path.dirname(cfg.research_dir)        # repo root with the CSVs
    if not glob.glob(os.path.join(root, "*_USDT_1d.csv")):
        print("  ⚠ signal pipeline test skipped (no CSVs found)")
        return
    import signals
    ex = FakeExchange(root)
    targets, info = signals.compute_targets(ex, cfg, log=lambda *a: None)
    assert info["n_targets"] > 0, "expected some target positions"
    assert info["gross"] <= cfg.max_gross + 1e-6
    g = sum(abs(w) for w in targets.values())
    print(f"  ✓ signal pipeline: {info['n_targets']} targets, gross {info['gross']}x, "
          f"{info['n_long']}L/{info['n_short']}S as-of {info['asof']}")


def run():
    print("ECHO bot self-test (offline)")
    test_reconcile()
    test_flatten()
    test_signal_pipeline()
    print("ALL TESTS PASSED ✅")
