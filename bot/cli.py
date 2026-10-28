"""Command line interface.

    python -m bot -c bot/config.toml run              # scheduler: trade once a day after the close
    python -m bot -c bot/config.toml once [--dry-run] # one cycle now
    python -m bot -c bot/config.toml targets          # today's target portfolio, no trading
    python -m bot -c bot/config.toml status           # equity, positions, halt state
    python -m bot -c bot/config.toml flatten --yes    # close every position and halt
    python -m bot -c bot/config.toml resume           # clear a risk halt after review
    python -m bot -c bot/config.toml replay --start 2025-01-01   # offline end-to-end replay (CSV)

Live trading additionally requires mode = "live" in the config, BINANCE_API_KEY /
BINANCE_API_SECRET in the environment, AND the --confirm-live flag on every command that trades.
"""
from __future__ import annotations
import argparse
import json
import logging
import logging.handlers
import os
import sys

import pandas as pd

from .binance import MAINNET, TESTNET, BinanceAPIError, BinanceFuturesClient
from .brokers import LiveBroker, PaperBroker
from .config import Config, load_config
from .data import BinanceData, CsvData
from .runner import Bot, StaleDataError
from .state import InstanceLock, Notifier, StateStore

log = logging.getLogger("bot")


def setup_logging(state_dir: str, verbose: bool):
    os.makedirs(state_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh = logging.StreamHandler(sys.stderr); sh.setFormatter(fmt)
    fh = logging.handlers.RotatingFileHandler(os.path.join(state_dir, "bot.log"), maxBytes=10_000_000, backupCount=5)
    fh.setFormatter(fmt)
    root.handlers[:] = [sh, fh]
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def make_client(cfg: Config, signed: bool) -> BinanceFuturesClient:
    url = cfg.exchange.base_url or (TESTNET if cfg.exchange.testnet else MAINNET)
    return BinanceFuturesClient(url, cfg.api_key if signed else "", cfg.api_secret if signed else "",
                                recv_window=cfg.exchange.recv_window)


def build(cfg: Config, state: dict, store: StateStore) -> Bot:
    notifier = Notifier(cfg.notify.webhook_url, cfg.notify.telegram_token, cfg.notify.telegram_chat_id,
                        prefix=f"[v4 {cfg.mode}{' testnet' if cfg.exchange.testnet else ''}]")
    if cfg.mode == "paper" and cfg.paper.data_source == "csv":
        data = CsvData(cfg.paper.csv_dir, cfg.strategy.exclude)
    else:
        client = make_client(cfg, signed=cfg.mode == "live")
        data = BinanceData(client, cfg.strategy.exclude, cfg.exchange.min_quote_volume_24h, cfg.exchange.max_workers)
    if cfg.mode == "paper":
        book = state.get("paper") or {"wallet": cfg.paper.initial_equity, "positions": {}}
        broker = PaperBroker(book, cfg.paper.fee, cfg.paper.slippage)
    else:
        broker = LiveBroker(data.client, data.filters(), cfg.exchange.margin_type,
                            cfg.exchange.exchange_leverage, state.setdefault("live_configured", {}))
        broker.prepare()
    return Bot(cfg, data, broker, store, state, notifier)


def _print_report(r):
    print(f"status: {r.status}  day: {r.day}  equity: {r.equity:,.2f}  {r.message}")
    if r.targets:
        print("target weights (pre-leverage):")
        for s, w in sorted(r.targets.items(), key=lambda kv: -abs(kv[1])):
            print(f"  {'LONG ' if w > 0 else 'SHORT'} {s:<16} {w * 100:+7.2f}%")
    for o in r.orders:
        print(f"  order {o.side:<4} {o.symbol:<16} qty {o.qty}  ~{float(o.qty) * o.price_hint:,.2f} USDT"
              f"  {'reduceOnly ' if o.reduce_only else ''}({o.reason})")
    for f in r.fills:
        print(f"  fill  {json.dumps(f, default=str)}")
    for s in r.skipped:
        print(f"  skip  {s}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bot", description="v4 Binance futures trading bot")
    ap.add_argument("-c", "--config", default=None, help="TOML config (default: built-in paper defaults)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--confirm-live", action="store_true", help="required for any live-trading command")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    o = sub.add_parser("once"); o.add_argument("--dry-run", action="store_true"); o.add_argument("--force", action="store_true")
    t = sub.add_parser("targets")
    for p in (o, t):
        p.add_argument("--as-of", help="offline/CSV only: act as if the bar of this UTC date (YYYY-MM-DD) just closed")
    sub.add_parser("status")
    f = sub.add_parser("flatten"); f.add_argument("--yes", action="store_true")
    sub.add_parser("resume")
    rp = sub.add_parser("replay"); rp.add_argument("--start", required=True); rp.add_argument("--end")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg.state_dir, args.verbose)

    if args.cmd == "replay":
        from .replay import replay
        if cfg.paper.csv_dir == ".":
            cfg.paper.csv_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        res = replay(cfg, args.start, args.end)
        print(f"replay {args.start}..{args.end or 'end'}: {res['days']} days, {res['orders']} orders, fees {res['fees']:,.2f}")
        for k in ("bot", "ideal"):
            s = res[k]
            print(f"  {k:<6} Sharpe {s['sharpe']:5.2f}  CAGR {s['cagr']:7.1%}  maxDD {s['max_dd']:6.1%}  total {s['total']:8.1%}")
        print(f"  daily-return correlation bot vs ideal: {res['corr']:.3f}")
        return 0

    trading = args.cmd in ("run", "once", "flatten") and not getattr(args, "dry_run", False)
    if cfg.mode == "live" and trading and not args.confirm_live:
        ap.error("live mode: add --confirm-live to trade real money")

    now = None
    if getattr(args, "as_of", None):
        if cfg.mode == "live" or cfg.paper.data_source != "csv":
            ap.error("--as-of is only for paper mode with data_source = csv")
        now = pd.Timestamp(args.as_of, tz="UTC") + pd.Timedelta(days=1, minutes=cfg.schedule.run_minute_utc)
    try:
        return _dispatch(ap, args, cfg, now)
    except (StaleDataError, BinanceAPIError, RuntimeError, ValueError) as e:
        log.error("%s: %s", type(e).__name__, e)
        return 2


def _dispatch(ap, args, cfg, now):
    store = StateStore(cfg.state_dir)
    with InstanceLock(cfg.state_dir):
        state = store.load()
        if args.cmd == "resume":
            state.update(halted=False, halt_reason="")
            peak = state.get("last_equity")
            if peak:
                state["peak_equity"] = peak                 # reset drawdown reference after review
            store.save(state); store.journal("resume")
            print("halt cleared; drawdown peak reset to last equity"); return 0
        bot = build(cfg, state, store)
        if args.cmd == "run":
            bot.run_forever()
        elif args.cmd == "once":
            _print_report(bot.run_cycle(now=now, dry_run=args.dry_run, force=args.force))
        elif args.cmd == "targets":
            _print_report(bot.run_cycle(now=now, dry_run=True))
        elif args.cmd == "status":
            prices = bot.data.mark_prices()
            eq = bot.broker.equity(prices)   # CSV source: priced at the last bar in the files
            print(f"mode {cfg.mode}  equity {eq:,.2f}  peak {state.get('peak_equity', eq):,.2f}  "
                  f"halted {state.get('halted', False)} {state.get('halt_reason', '')}  last cycle {state.get('last_cycle_day')}")
            for s, q in sorted(bot.broker.positions().items()):
                px = prices.get(s, 0.0)
                print(f"  {s:<16} qty {q:+.6g}  notional {q * px:+,.2f}")
        elif args.cmd == "flatten":
            if not args.yes:
                ap.error("flatten closes every position; add --yes")
            prices = bot.data.mark_prices()
            fills = bot.halt("manual flatten", prices)
            print(json.dumps(fills, indent=2, default=str))
    return 0
