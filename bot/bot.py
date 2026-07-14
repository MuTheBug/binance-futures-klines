"""
ECHO Engine autonomous trading bot — orchestrator.

Single-threaded event loop:
  * long-polls Telegram for commands and dispatches them,
  * once per day at REBALANCE_UTC_HOUR:MINUTE, recomputes the ECHO target weights and
    reconciles the live book toward them,
  * enforces a max-drawdown circuit breaker (flatten + pause + alert) on every rebalance.

Safe by default: DRY_RUN and TESTNET on until you explicitly turn them off.
"""
from __future__ import annotations
import time
import datetime as dt
import traceback

from config import Config
from exchange import make_exchange, BinanceError
from notifier import Telegram
from state import State
import signals
import trader


def _utcnow():
    return dt.datetime.now(dt.timezone.utc)


def pct(x):
    return f"{x*100:+.1f}%"


class EchoBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ex = make_exchange(cfg)
        self.tg = Telegram(cfg.tg_token, cfg.tg_chat_ids)
        self.state = State(cfg.state_path)
        self.tg.offset = self.state["tg_offset"]
        self.last_targets = {}
        self.last_info = {}
        self.log_lines = []
        self.live_blocked = False        # set if account is in hedge mode (unsafe for this bot)

    # ----------------------------- logging -----------------------------
    def log(self, msg):
        line = f"{_utcnow().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.log_lines.append(line)
        self.log_lines = self.log_lines[-200:]

    def alert(self, text):
        self.log("ALERT: " + text.replace("\n", " | ")[:200])
        self.tg.send(text)

    # ----------------------------- account helpers -----------------------------
    def equity(self):
        if not self.cfg.api_key:
            return 0.0
        return self.ex.balance_usdt()

    # ----------------------------- rebalance -----------------------------
    def do_rebalance(self, force=False, source="schedule"):
        cfg = self.cfg
        if self.state["stopped_out"] and not force:
            self.log("rebalance skipped — circuit breaker latched (use /resume to clear)")
            return
        if self.state["paused"] and not force:
            self.log("rebalance skipped — paused")
            return
        self.log(f"=== REBALANCE ({source}) {cfg.summary()} ===")
        self.alert(f"⚙️ Rebalancing ({source}) — {cfg.summary()}")
        if cfg.api_key:
            self.ex.sync_time()              # align clock before signed calls
        try:
            targets, info = signals.compute_targets(self.ex, cfg, log=self.log)
        except Exception as e:
            self.alert(f"❌ signal error: {e}")
            self.log(traceback.format_exc())
            return
        self.last_targets, self.last_info = targets, info
        self.log(f"targets: {info}")

        if not cfg.api_key:
            self.alert("ℹ️ No API key set — computed targets only (not trading).\n" +
                       self._fmt_targets(targets, info))
            self.state["last_rebalance_date"] = _utcnow().strftime("%Y-%m-%d")
            return

        try:
            equity = self.equity()
            positions = self.ex.positions()
            prices = self.ex.prices()
        except BinanceError as e:
            self.alert(f"❌ account/market fetch error: {e}")
            return

        # circuit breaker BEFORE trading
        self.state.record_equity(equity)
        dd = self.state.drawdown(equity)
        if dd <= -cfg.max_drawdown_stop:
            self.alert(f"🛑 CIRCUIT BREAKER: drawdown {pct(dd)} ≤ -{cfg.max_drawdown_stop:.0%}. "
                       f"Flattening and pausing.")
            trader.flatten(self.ex, cfg, log=self.log)
            self.state["stopped_out"] = True
            self.state["paused"] = True
            return

        # soft profit-trailing throttle (backtested in research/trailing.py):
        # halve the book past trail_dd drawdown, restore after half-recovery.
        # Sits below the hard circuit breaker; disable with TRAIL_DD=0.
        if cfg.trail_dd > 0:
            throttled = bool(self.state["throttled"])
            if throttled and dd >= -cfg.trail_dd / 2:
                self.state["throttled"] = False
                self.alert(f"🟢 trailing throttle OFF (drawdown recovered to {pct(dd)})")
            elif not throttled and dd <= -cfg.trail_dd:
                self.state["throttled"] = True
                self.alert(f"🟠 trailing throttle ON: drawdown {pct(dd)} ≤ -{cfg.trail_dd:.0%} — "
                           f"book scaled x{cfg.trail_throttle:g} until DD < {cfg.trail_dd / 2:.0%}")
            if self.state["throttled"]:
                targets = {s: w * cfg.trail_throttle for s, w in targets.items()}

        if self.live_blocked and not cfg.dry_run:
            self.alert("⛔ Account is in HEDGE mode — refusing live orders (this bot needs "
                       "One-way mode). Switch Position Mode to One-way, then /resume.\n" +
                       self._fmt_targets(targets, info))
            return
        orders = trader.reconcile(targets, equity, positions, prices, self.ex, cfg)
        if not orders:
            self.alert(f"✅ Already on target. Equity ${equity:,.2f} DD {pct(dd)}.\n" +
                       self._fmt_targets(targets, info))
        else:
            res = trader.execute(orders, self.ex, cfg, log=self.log)
            head = "🧪 DRY-RUN orders" if cfg.dry_run else "📈 Orders sent"
            self.alert(f"{head} ({res['n']}) — equity ${equity:,.2f} DD {pct(dd)}\n" +
                       self._fmt_orders(orders) + "\n" + self._fmt_targets(targets, info) +
                       (f"\n⚠️ {len(res['errors'])} errors" if res["errors"] else ""))
        self.state["last_rebalance_date"] = _utcnow().strftime("%Y-%m-%d")

    # ----------------------------- formatting -----------------------------
    def _fmt_targets(self, targets, info):
        if not targets:
            return "no target positions"
        rows = sorted(targets.items(), key=lambda kv: -abs(kv[1]))
        body = "\n".join(f"  {'L' if w>0 else 'S'} {s:<12} {w*100:+5.1f}%" for s, w in rows)
        return (f"<b>targets {info.get('asof','')}</b> gross {info.get('gross')}x "
                f"({info.get('n_long')}L/{info.get('n_short')}S of {info.get('eligible')} elig)\n{body}")

    def _fmt_orders(self, orders):
        return "\n".join(f"  {o.side:4} {o.symbol:<12} ~${o.notional:,.0f} ({o.reason})"
                         for o in orders[:30])

    # ----------------------------- commands -----------------------------
    def handle_command(self, chat_id, text):
        parts = text.split()
        cmd = parts[0].lstrip("/").split("@")[0].lower()
        args = parts[1:]
        cfg = self.cfg
        try:
            if cmd in ("start", "help"):
                self.tg.send(self._help(), chat_id)
            elif cmd == "id":
                self.tg.send(f"chat_id: <code>{chat_id}</code>", chat_id)
            elif cmd == "status":
                self.tg.send(self._status(), chat_id)
            elif cmd == "risk":
                self.tg.send(self._risk(), chat_id)
            elif cmd == "balance":
                eq = self.equity()
                self.tg.send(f"💰 equity: ${eq:,.2f}  peak: ${self.state['peak_equity'] or 0:,.2f}  "
                             f"DD: {pct(self.state.drawdown(eq))}", chat_id)
            elif cmd == "positions":
                self.tg.send(self._positions(), chat_id)
            elif cmd == "pnl":
                self.tg.send(self._pnl(), chat_id)
            elif cmd == "report":
                self.tg.send(self._report(), chat_id)
            elif cmd == "weights":
                self.tg.send(self._fmt_targets(self.last_targets, self.last_info)
                             if self.last_targets else "no targets computed yet — use /signal", chat_id)
            elif cmd == "signal":
                self.tg.send("computing signal…", chat_id)
                targets, info = signals.compute_targets(self.ex, cfg, log=self.log)
                self.last_targets, self.last_info = targets, info
                self.tg.send(self._fmt_targets(targets, info), chat_id)
            elif cmd == "rebalance":
                self.tg.send("forcing rebalance…", chat_id)
                self.do_rebalance(force=True, source="manual")
            elif cmd == "pause":
                self.state["paused"] = True
                self.tg.send("⏸️ paused — no scheduled rebalances", chat_id)
            elif cmd == "resume":
                self.state["paused"] = False
                self.state["stopped_out"] = False
                self.tg.send("▶️ resumed (circuit breaker cleared)", chat_id)
            elif cmd == "flat":
                self.tg.send("flattening all positions…", chat_id)
                res = trader.flatten(self.ex, cfg, log=self.log)
                self.tg.send(f"closed {res['n']} positions"
                             + (" (dry-run)" if cfg.dry_run else ""), chat_id)
            elif cmd == "leverage":
                if args:
                    cfg.leverage = max(0.0, min(cfg.max_gross, float(args[0])))
                self.tg.send(f"leverage = {cfg.leverage:g}x (cap {cfg.max_gross:g}x)", chat_id)
            elif cmd == "dryrun":
                if args:
                    cfg.dry_run = args[0].lower() in ("on", "1", "true", "yes")
                    self.ex.dry_run = cfg.dry_run
                self.tg.send(f"DRY_RUN = {cfg.dry_run}", chat_id)
            elif cmd == "log":
                self.tg.send("<pre>" + "\n".join(self.log_lines[-15:]) + "</pre>", chat_id)
            else:
                self.tg.send(f"unknown command: /{cmd}\n" + self._help(), chat_id)
        except Exception as e:
            self.tg.send(f"❌ /{cmd} error: {e}", chat_id)
            self.log(traceback.format_exc())

    def _help(self):
        return ("<b>ECHO Engine bot</b>\n"
                "/status – bot + account snapshot\n"
                "/balance – equity, peak, drawdown\n"
                "/positions – open positions + uPnL\n"
                "/pnl – PnL / equity summary\n"
                "/report – live performance vs backtest\n"
                "/signal – recompute target weights (no trade)\n"
                "/weights – last computed targets\n"
                "/rebalance – force a rebalance now\n"
                "/flat – close ALL positions (panic)\n"
                "/pause /resume – stop/start auto-trading\n"
                "/leverage &lt;x&gt; – set leverage multiplier\n"
                "/dryrun &lt;on|off&gt; – toggle paper mode\n"
                "/risk – risk settings\n/log – recent log\n/id – chat id")

    def _status(self):
        cfg = self.cfg
        eq = self.equity()
        nxt = self._next_rebalance()
        flags = []
        if cfg.dry_run: flags.append("DRY-RUN")
        flags.append("TESTNET" if cfg.testnet else "MAINNET")
        if self.state["paused"]: flags.append("PAUSED")
        if self.state["stopped_out"]: flags.append("STOPPED-OUT")
        ddtxt = pct(self.state.drawdown(eq)) if eq > 0 else "n/a"
        return (f"<b>ECHO bot</b> [{' '.join(flags)}]\n"
                f"equity: ${eq:,.2f}  DD: {ddtxt}\n"
                f"lev {cfg.leverage:g}x · vol {cfg.target_vol:g} · blend "
                f"{cfg.blend_trend:g}/{1-cfg.blend_trend:g} · maxPos {cfg.max_positions}\n"
                f"targets: {self.last_info.get('n_targets','-')} "
                f"(gross {self.last_info.get('gross','-')}x, sleeves {self.last_info.get('sleeves','-')})\n"
                f"next rebalance: {nxt} UTC")

    def _risk(self):
        cfg = self.cfg
        return (f"<b>risk</b>\nmax gross {cfg.max_gross:g}x · DD stop "
                f"{cfg.max_drawdown_stop:.0%} · min order ${cfg.min_order_usdt:g}\n"
                f"no-churn band {cfg.rebalance_band:.0%} · binance lev {cfg.binance_leverage}x\n"
                f"min $vol ${cfg.min_dollar_vol:,.0f} · min history {cfg.min_history_days}d")

    def _positions(self):
        if not self.cfg.api_key:
            return "no API key set"
        det = self.ex.position_detail()
        if not det:
            return "no open positions"
        det.sort(key=lambda d: -abs(d["amt"] * d["mark"]))
        tot = sum(d["upnl"] for d in det)
        body = "\n".join(f"  {'L' if d['amt']>0 else 'S'} {d['symbol']:<12} "
                         f"${abs(d['amt']*d['mark']):,.0f}  uPnL ${d['upnl']:+,.2f}" for d in det)
        return f"<b>positions</b> ({len(det)})  total uPnL ${tot:+,.2f}\n{body}"

    def _pnl(self):
        hist = self.state["equity_history"] or []
        eq = self.equity()
        peak = self.state["peak_equity"] or 0.0
        first = hist[0][1] if hist else eq
        tot = (eq / first - 1.0) if first else 0.0
        return (f"<b>PnL</b>\nequity ${eq:,.2f}  peak ${peak:,.2f}\n"
                f"since start ({hist[0][0][:10] if hist else 'n/a'}): {pct(tot)}\n"
                f"drawdown: {pct(self.state.drawdown(eq))}  samples: {len(hist)}")

    def _report(self):
        hist = self.state["equity_history"] or []
        if len(hist) < 2:
            return "not enough history yet for a report (need a few days of equity samples)"
        eq = self.equity() or hist[-1][1]
        first = hist[0][1]
        try:
            t0 = dt.datetime.strptime(hist[0][0], "%Y-%m-%dT%H:%M:%SZ")
            t1 = dt.datetime.strptime(hist[-1][0], "%Y-%m-%dT%H:%M:%SZ")
            days = max(1.0, (t1 - t0).total_seconds() / 86400.0)
        except ValueError:
            days = float(len(hist))
        tot = (eq / first - 1.0) if first else 0.0
        dd = self.state.drawdown(eq)
        # only annualize once there is a meaningful span (>=14d), else show the daily pace
        ann_line = ""
        if days >= 14 and first > 0 and eq > 0:
            ann = (eq / first) ** (365.0 / days) - 1.0
            ann_line = f"  ann.pace {pct(ann)}"
        return (f"<b>live report</b> ({days:.0f}d)\n"
                f"equity ${eq:,.2f} (start ${first:,.2f})  since {hist[0][0][:10]}\n"
                f"total {pct(tot)}{ann_line}  drawdown {pct(dd)}\n"
                f"<i>backtest ref @2x: median ~+7.5%/mo, expect −40/−50% DD en route. "
                f"Live below backtest is normal.</i>")

    def _maybe_summary(self):
        cfg = self.cfg
        now = _utcnow()
        today = now.strftime("%Y-%m-%d")
        if now.hour < cfg.summary_utc_hour or self.state["last_summary_date"] == today:
            return
        self.state["last_summary_date"] = today
        try:
            self.alert("📅 Daily check-in\n" + self._status() + "\n" + self._positions())
        except Exception as e:
            self.log(f"summary error: {e}")

    # ----------------------------- scheduling / loop -----------------------------
    def _days_since_last(self, now):
        last = self.state["last_rebalance_date"]
        if not last:
            return 10 ** 6
        try:
            d = dt.datetime.strptime(last, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            return 10 ** 6
        return (now.date() - d.date()).days

    def _next_rebalance(self):
        now = _utcnow()
        run = now.replace(hour=self.cfg.rebalance_utc_hour, minute=self.cfg.rebalance_utc_minute,
                          second=0, microsecond=0)
        every = self.cfg.rebalance_every_days
        # slot today still ahead and cadence satisfied -> today; else next eligible day
        if now < run and self._days_since_last(now) >= every:
            nxt = run
        else:
            wait = max(1, every - self._days_since_last(now)) if now >= run else 1
            nxt = run + dt.timedelta(days=wait)
        return nxt.strftime("%Y-%m-%d %H:%M")

    def _due(self):
        now = _utcnow()
        run = now.replace(hour=self.cfg.rebalance_utc_hour, minute=self.cfg.rebalance_utc_minute,
                          second=0, microsecond=0)
        today = now.strftime("%Y-%m-%d")
        return (now >= run and self.state["last_rebalance_date"] != today
                and self._days_since_last(now) >= self.cfg.rebalance_every_days)

    def tick(self):
        if not self.state["paused"] and not self.state["stopped_out"] and self._due():
            self.do_rebalance(source="schedule")
        self._maybe_summary()

    def run(self):
        cfg = self.cfg
        try:
            self.ex.ping(); self.ex.sync_time()
            conn = f"ok (clock offset {self.ex.time_offset}ms)"
        except Exception as e:
            conn = f"FAILED ({e})"
        warn = ""
        if cfg.api_key:
            try:
                if self.ex.hedge_mode():
                    self.live_blocked = True
                    warn = "\n⛔ Account is in HEDGE mode — live orders blocked. Switch Binance " \
                           "Position Mode to One-way to trade."
            except Exception as e:
                warn = f"\n⚠️ could not verify position mode: {e}"
        self.alert(f"🟢 ECHO bot online — {cfg.summary()} | binance ping: {conn}{warn}\n"
                   f"send /help for commands")
        self.log("entering main loop")
        while True:
            try:
                for chat_id, text in self.tg.poll():
                    if text.startswith("/"):
                        self.handle_command(chat_id, text)
                if self.tg.offset is not None:
                    self.state["tg_offset"] = self.tg.offset
                self.tick()
                if not self.tg.enabled:
                    time.sleep(cfg.poll_seconds)
            except KeyboardInterrupt:
                self.alert("🔴 ECHO bot stopping (KeyboardInterrupt)")
                break
            except Exception as e:
                self.log("loop error: " + str(e))
                self.log(traceback.format_exc())
                time.sleep(cfg.poll_seconds)
