"""The trading cycle and the scheduler loop.

One cycle (once per day, a few minutes after the 00:00 UTC daily close):
  1. guard rails: STOP file, halted flag, already-traded-today (idempotent restarts)
  2. data: universe, CLOSED daily bars, staleness check, current prices
  3. paper only: apply funding and simulated catastrophe stops since the last cycle
  4. risk: equity vs peak (max drawdown) and vs last cycle (max cycle loss) -> flatten + halt
  5. strategy -> target weights -> exchange-valid order plan (reduce-only first)
  6. execute, place optional live stops, persist state, journal, notify
"""
from __future__ import annotations
import logging
import signal
import time
from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from .data import last_closed_day
from .execution import plan_orders
from .state import Notifier, StateStore
from .strategy import compute_targets

log = logging.getLogger(__name__)


class StaleDataError(RuntimeError):
    pass


@dataclass
class CycleReport:
    status: str
    day: str = ""
    equity: float = 0.0
    orders: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    targets: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)
    message: str = ""


def check_risk(state: dict, equity: float, risk) -> str | None:
    peak = max(float(state.get("peak_equity", equity)), equity)
    state["peak_equity"] = peak
    dd = 1.0 - equity / peak if peak > 0 else 0.0
    if dd >= risk.max_drawdown:
        return f"max drawdown hit: equity {equity:.2f} is {dd:.1%} below peak {peak:.2f}"
    last = state.get("last_equity")
    if last and equity / float(last) - 1.0 <= -risk.max_cycle_loss:
        return f"max cycle loss hit: equity {equity:.2f} vs {float(last):.2f} last cycle"
    return None


class Bot:
    def __init__(self, cfg: Config, data, broker, store: StateStore, state: dict,
                 notifier: Notifier | None = None, clock=None, sleep=time.sleep):
        self.cfg, self.data, self.broker, self.store, self.state = cfg, data, broker, store, state
        self.notifier = notifier or Notifier()
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self.sleep = sleep
        self._stop = False
        if state.get("mode", cfg.mode) != cfg.mode:
            raise RuntimeError(f"state dir {store.dir} belongs to mode {state['mode']!r}, config says {cfg.mode!r}")
        state.setdefault("mode", cfg.mode)
        state.setdefault("run_counter", 0)

    # ----------------------------------------------------------------- helpers
    def save(self):
        if self.broker.mode == "paper":
            self.state["paper"] = self.broker.book
        self.store.save(self.state)

    def halt(self, reason: str, prices: dict, flatten: bool = True) -> list:
        log.critical("HALT: %s", reason)
        fills = self.broker.flatten(prices, tag=f"halt{int(time.time())}") if flatten else []
        self.state.update(halted=True, halt_reason=reason)
        self.store.journal("halt", reason=reason, fills=fills)
        self.save()
        self.notifier.send(f"HALTED and flattened: {reason}. Run `resume` after review.")
        return fills

    def _prices(self, bars) -> dict[str, float]:
        if hasattr(self.data, "all"):                          # CsvData: trade at the bar close
            return self.data.mark_prices(bars.last_day)
        return self.data.mark_prices()

    # ----------------------------------------------------------------- one cycle
    def run_cycle(self, now: pd.Timestamp | None = None, dry_run=False, force=False) -> CycleReport:
        cfg, st = self.cfg, self.state
        now = now or self.clock()
        day = last_closed_day(now)
        dstr = str(day.date())
        if self.store.stop_requested():
            return CycleReport("paused", dstr, message=f"STOP file present ({self.store.stop_path})")
        if st.get("halted"):
            return CycleReport("halted", dstr, message=st.get("halt_reason", ""))
        if not force and not dry_run and st.get("last_cycle_day") == dstr:
            return CycleReport("already-done", dstr)

        hedge = cfg.strategy.hedge_symbol
        universe = self.data.universe(must_include=[hedge])
        bars = self.data.daily_bars(universe, cfg.strategy.history_bars, now)
        if bars.last_day != day or pd.isna(bars.close[hedge].iloc[-1]):
            raise StaleDataError(f"latest closed bar is {bars.last_day.date()}, expected {dstr}")
        if bars.close.iloc[-1].notna().sum() < 20:
            raise StaleDataError("fewer than 20 symbols have a bar for the latest day")
        prices = self._prices(bars)
        if not prices.get(hedge):
            raise StaleDataError(f"no price for {hedge}")

        if self.broker.mode == "paper" and not dry_run:
            last = st.get("last_cycle_day")
            if last:
                stops = self.broker.simulate_stops(bars, pd.Timestamp(last, tz="UTC"), cfg.risk.stop_loss_pct)
                for s in stops:
                    self.store.journal("paper_stop", **s)
                if hasattr(self.data, "funding_paid") and self.broker.positions():
                    start = int(st.get("last_cycle_ts_ms", 0)) or int((now - pd.Timedelta(days=1)).timestamp() * 1000)
                    rates = self.data.funding_paid(list(self.broker.positions()), start, int(now.timestamp() * 1000))
                    paid = self.broker.apply_funding(rates, prices)
                    self.store.journal("paper_funding", paid=paid)

        out = compute_targets(bars.close, bars.volume, cfg.strategy)
        equity = self.broker.equity(prices)
        positions = self.broker.positions()

        if not dry_run:
            reason = check_risk(st, equity, cfg.risk)
            if reason:
                fills = self.halt(reason, prices)
                return CycleReport("halted", dstr, equity, fills=fills, message=reason)
            if self.broker.mode == "live":
                foreign = set(positions) - set(st.get("owned", []))
                if foreign and not cfg.exchange.adopt_existing_positions:
                    raise RuntimeError(f"account has positions the bot did not open: {sorted(foreign)}. "
                                       "Use a dedicated (sub)account or set exchange.adopt_existing_positions")

        plan = plan_orders(out.weights, equity, positions, prices, self.data.filters(), cfg.risk)
        report = CycleReport("dry-run" if dry_run else "done", dstr, equity, orders=plan.orders,
                             targets={k: round(float(v), 6) for k, v in out.weights.items()},
                             skipped=plan.skipped)
        if dry_run:
            return report

        st["run_counter"] += 1
        run = st["run_counter"]
        self.save()                      # persist before sending orders: clientOrderIds stay unique after a crash
        self.store.journal("cycle_start", day=dstr, equity=equity, targets=report.targets,
                           pump_names=out.pump_names, n_eligible=out.n_eligible,
                           gross_scale=plan.gross_scale, skipped=plan.skipped)
        for i, o in enumerate(plan.orders):
            if self._stop:
                log.warning("shutdown requested mid-cycle; remaining orders not sent")
                break
            fill = self.broker.execute(o, tag=f"{day:%y%m%d}-{run}-{i}")
            report.fills.append(fill)
            self.store.journal("fill", **fill)
            if fill.get("status") == "FILLED":            # persist ownership immediately (crash safety)
                st["owned"] = sorted(set(st.get("owned", [])) | {o.symbol})
                self.save()
            if fill.get("status") not in ("FILLED", "NEW", "PARTIALLY_FILLED"):
                log.error("order not filled: %s", fill)
        if self.broker.mode == "live" and cfg.risk.stop_loss_pct > 0:
            for s in self.broker.place_stops(cfg.risk.stop_loss_pct, tag=f"{day:%y%m%d}-{run}"):
                self.store.journal("stop", **s)

        after = self.broker.positions()
        eq_after = self.broker.equity(prices)
        gross = sum(abs(q) * prices.get(s, 0.0) for s, q in after.items())
        st.update(last_cycle_day=dstr, last_cycle_ts_ms=int(now.timestamp() * 1000),
                  last_equity=eq_after, owned=sorted(after))
        self.save()
        self.store.record_equity(dstr, eq_after, gross, len(after))
        self.store.journal("cycle_end", day=dstr, equity=eq_after, gross=gross, positions=after)
        report.equity = eq_after
        rejected = [f for f in report.fills if f.get("status") == "REJECTED"]
        self.notifier.send(f"{dstr}: equity {eq_after:,.2f}, {len(after)} positions, gross {gross / max(eq_after, 1e-9):.2f}x, "
                           f"{len(report.fills)} orders" + (f", {len(rejected)} REJECTED" if rejected else ""))
        return report

    # ----------------------------------------------------------------- scheduler
    def _due(self, now) -> bool:
        target = now.floor("D") + pd.Timedelta(minutes=self.cfg.schedule.run_minute_utc)
        return now >= target and self.state.get("last_cycle_day") != str(last_closed_day(now).date())

    def run_forever(self):
        def _sig(signum, _):
            log.warning("signal %s received; stopping after the current step", signum)
            self._stop = True
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, _sig)
        log.info("scheduler started (mode=%s, runs %d min after 00:00 UTC)", self.cfg.mode,
                 self.cfg.schedule.run_minute_utc)
        failures, fail_day = 0, None
        while not self._stop:
            now = self.clock()
            day = str(last_closed_day(now).date())
            if self.state.get("halted") or self.store.stop_requested() or not self._due(now) or fail_day == day:
                self.sleep(30)
                continue
            try:
                r = self.run_cycle(now)
                log.info("cycle %s: %s %s", r.day, r.status, r.message)
                failures = 0
            except Exception as e:
                failures += 1
                log.exception("cycle failed (%d/%d)", failures, self.cfg.schedule.max_retries)
                self.store.journal("cycle_error", day=day, error=repr(e), attempt=failures)
                if failures >= self.cfg.schedule.max_retries:
                    self.notifier.send(f"{day}: cycle failed {failures}x, skipping today: {e!r}")
                    fail_day, failures = day, 0
                else:
                    self.notifier.send(f"{day}: cycle error (attempt {failures}), retrying: {e!r}")
                    self.sleep(60 * self.cfg.schedule.retry_minutes)
        log.info("scheduler stopped")
