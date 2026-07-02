#!/usr/bin/env python3
"""Entry point for the ECHO Engine trading bot.

Usage:
    python3 run.py            # start the bot (Telegram + daily auto-rebalance)
    python3 run.py --signal   # compute & print target weights once, then exit (no trading)
    python3 run.py --selftest # offline logic test (no network/keys needed)
"""
from __future__ import annotations
import sys
from config import Config


def main():
    cfg = Config.load()
    if "--selftest" in sys.argv:
        import selftest
        selftest.run()
        return
    if "--signal" in sys.argv:
        from exchange import BinanceFutures
        import signals
        ex = BinanceFutures(cfg.api_key, cfg.api_secret, testnet=cfg.testnet, dry_run=True)
        targets, info = signals.compute_targets(ex, cfg)
        print("\nINFO:", info)
        for s, w in sorted(targets.items(), key=lambda kv: -abs(kv[1])):
            print(f"  {'LONG ' if w>0 else 'SHORT'} {s:<14} {w*100:+6.2f}%")
        return
    from bot import EchoBot
    EchoBot(cfg).run()


if __name__ == "__main__":
    main()
