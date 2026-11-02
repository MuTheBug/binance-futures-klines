"""Grid bot command line.

    python -m gridbot -c gridbot/config.toml plan          # what grid would be placed now (no trading)
    python -m gridbot -c gridbot/config.toml once          # one cycle
    python -m gridbot -c gridbot/config.toml run           # loop every poll_seconds
    python -m gridbot -c gridbot/config.toml status
    python -m gridbot -c gridbot/config.toml flatten --yes # cancel grid orders, close positions, halt
    python -m gridbot -c gridbot/config.toml resume        # clear a halt after review
    python -m gridbot -c gridbot/config.toml replay --symbol DOGEUSDT --start 2025-01-01 [--end ...]

Live trading needs mode = "live", BINANCE_API_KEY / BINANCE_API_SECRET, and --confirm-live.
"""
from __future__ import annotations
import argparse
import json
import logging
import logging.handlers
import os
import sys

from bot.binance import MAINNET, TESTNET, BinanceAPIError, BinanceFuturesClient, parse_filters
from bot.state import InstanceLock, Notifier, StateStore
from .config import Config, load_config
from .feeds import LivePrices, PaperFeed
from .runner import GridBot
from .sim_exchange import SimExchange
from .strategy import GridStrategy

log = logging.getLogger("gridbot")


def setup_logging(state_dir, verbose):
    os.makedirs(state_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh = logging.StreamHandler(sys.stderr); sh.setFormatter(fmt)
    fh = logging.handlers.RotatingFileHandler(os.path.join(state_dir, "gridbot.log"), maxBytes=10_000_000, backupCount=5)
    fh.setFormatter(fmt)
    root.handlers[:] = [sh, fh]
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def build(cfg: Config, state: dict, store: StateStore, client=None):
    url = cfg.exchange.base_url or (TESTNET if cfg.exchange.testnet else MAINNET)
    live = cfg.mode == "live"
    client = client or BinanceFuturesClient(url, cfg.api_key if live else "", cfg.api_secret if live else "",
                                            recv_window=cfg.exchange.recv_window)
    all_f = parse_filters(client.exchange_info())
    missing = [s for s in cfg.symbols if s not in all_f]
    if missing:
        raise ValueError(f"not tradable USDT perpetual(s) on this exchange: {missing}")
    filters = {s: all_f[s] for s in cfg.symbols}
    notifier = Notifier(cfg.notify.webhook_url, cfg.notify.telegram_token, cfg.notify.telegram_chat_id,
                        prefix=f"[grid {cfg.mode}{' testnet' if cfg.exchange.testnet else ''}]")
    strategy = GridStrategy(cfg.grid, lambda s, i, n: [k for k in client.klines(s, i, limit=n + 1)][:-1])
    if live:
        client.sync_time()
        if client.dual_side_position():
            raise RuntimeError("account is in Hedge Mode; switch Binance futures to One-way Mode")
        for s in cfg.symbols:
            try:
                client.set_margin_type(s, cfg.risk.margin_type)
            except BinanceAPIError as e:
                log.warning("%s margin type unchanged: %s", s, e)
            client.set_leverage(s, cfg.risk.exchange_leverage)
        feed, exchange, sim = LivePrices(client, cfg.symbols), client, None
        owned = {p["symbol"] for p in client.position_risk() if float(p.get("positionAmt", 0)) != 0} & set(cfg.symbols)
        if owned and not state.get("grids"):
            raise RuntimeError(f"open positions in {sorted(owned)} that the bot did not create; "
                               "use a dedicated sub-account or close them first")
    else:
        sim = SimExchange(filters, cfg.paper.initial_balance, leverage=cfg.risk.exchange_leverage,
                          maker_fee=cfg.paper.maker_fee, taker_fee=cfg.paper.taker_fee,
                          slippage=cfg.paper.slippage, through=cfg.paper.through)
        if state.get("sim"):
            sim.load_dict(state["sim"])
        feed, exchange = PaperFeed(client, sim, cfg.symbols, state), sim
    bot = GridBot(cfg, exchange, filters, feed, strategy, store, state, notifier, sim=sim)
    return bot, client, filters, strategy


def cmd_plan(cfg, bot, client, filters, strategy):
    equity = bot.equity()
    prices = {p["symbol"]: float(p["price"]) for p in client.ticker_price()}
    print(f"equity {equity:,.2f} USDT, exposure {cfg.risk.exposure}x, levels/side {cfg.grid.levels}")
    for s, w in cfg.weights().items():
        p = strategy.plan(s)
        f = filters[s]
        if not p.allowed:
            print(f"  {s}: NO GRID ({p.reason})")
            continue
        g = bot.engine.new_grid(s, prices[s], p, equity * w, 0)
        if g is None:
            print(f"  {s}: capital {equity * w:.2f} too small: one level needs >= {f.min_notional * 1.02:.2f} USDT")
            continue
        lo, hi = g.level(-g.n), g.level(g.n)
        print(f"  {s}: price {prices[s]:.6g} | {p.reason} | n={g.n}/side, {g.q_quote:.2f} USDT per level "
              f"(exchange min {f.min_notional}), range {lo:.6g}..{hi:.6g} ({(hi / lo - 1):.1%}), "
              f"max inventory {g.max_notional():.2f} USDT")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gridbot", description="Binance USDT-M grid bot")
    ap.add_argument("-c", "--config")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--confirm-live", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "once", "plan", "status", "resume"):
        sub.add_parser(c)
    fl = sub.add_parser("flatten"); fl.add_argument("--yes", action="store_true")
    rp = sub.add_parser("replay")
    rp.add_argument("--symbol", required=True); rp.add_argument("--start", required=True); rp.add_argument("--end")
    rp.add_argument("--balance", type=float, default=100.0)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.state_dir, args.verbose)

    if args.cmd == "replay":
        from .replay_cli import replay_cmd
        return replay_cmd(cfg, args)
    if cfg.mode == "live" and args.cmd in ("run", "once", "flatten") and not args.confirm_live:
        ap.error("live mode: add --confirm-live to trade real money")
    store = StateStore(cfg.state_dir)
    try:
        with InstanceLock(cfg.state_dir):
            state = store.load()
            if args.cmd == "resume":
                state.update(halted=False, halt_reason="")
                state.pop("peak_equity", None); state.pop("day", None)
                store.save(state); store.journal("resume")
                print("halt cleared; drawdown peak reset"); return 0
            bot, client, filters, strategy = build(cfg, state, store)
            if args.cmd == "run":
                bot.run_forever()
            elif args.cmd == "once":
                out = bot.cycle()
                print(json.dumps({k: v for k, v in out.items() if k != "steps"}, default=str))
                for r in out.get("steps", []):
                    print(f"  {r.symbol}: {r.action} placed {r.placed} cancelled {r.cancelled} "
                          f"market {r.market_trades} errors {r.errors}")
            elif args.cmd == "plan":
                cmd_plan(cfg, bot, client, filters, strategy)
            elif args.cmd == "status":
                eq = bot.equity()
                print(f"mode {cfg.mode} equity {eq:,.2f} peak {state.get('peak_equity', eq):,.2f} "
                      f"halted {state.get('halted', False)} {state.get('halt_reason', '')}")
                for p in bot.ex.position_risk():
                    print(f"  position {p['symbol']}: {float(p['positionAmt']):+.6g} @ {float(p['entryPrice']):.6g}")
                for s, g in state.get("grids", {}).items():
                    n_orders = len([o for o in bot.ex.open_orders(s) if o.get("clientOrderId", "").startswith(f"g{g['gid']}-")])
                    print(f"  grid {s}: centre {g['centre']:.6g} g {g['g']:.3%} n {g['n']} bias {g['bias']} "
                          f"{g['q_quote']:.2f} USDT/level, {n_orders} resting orders")
            elif args.cmd == "flatten":
                if not args.yes:
                    ap.error("flatten closes every position; add --yes")
                bot.halt("manual flatten")
    except (BinanceAPIError, RuntimeError, ValueError) as e:
        log.error("%s: %s", type(e).__name__, e)
        return 2
    return 0
