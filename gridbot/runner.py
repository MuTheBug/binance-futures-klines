"""Grid bot main loop: prices -> risk checks -> one engine step per symbol -> persist.

Risk layer (on top of each grid's own exit rule):
  * STOP file in the state dir pauses trading (orders stay as they are).
  * account drawdown from peak > max_drawdown, or loss since 00:00 UTC > max_daily_loss
    -> cancel every grid order, flatten every symbol, halt until `resume`.
  * more than max_resets_per_day range exits on a symbol -> flatten it and cool down (whipsaw guard).
  * optional periodic re-centre (recenter_hours) and daily regime check (trend filter).
"""
from __future__ import annotations
import logging
import signal
import time

from .engine import GridEngine, StepResult
from .state_io import Journal

log = logging.getLogger(__name__)
DAY_MS = 86_400_000


class GridBot:
    def __init__(self, cfg, exchange, filters, feed, strategy, store, state, notifier, sim=None,
                 clock=lambda: int(time.time() * 1000), sleep=time.sleep):
        self.cfg, self.ex, self.filters, self.feed, self.strategy = cfg, exchange, filters, feed, strategy
        self.store, self.state, self.notifier, self.sim = store, state, notifier, sim
        self.clock, self.sleep = clock, sleep
        self.journal = Journal(store)
        g, r = cfg.grid, cfg.risk
        self.engine = GridEngine(exchange, filters, state, n=g.levels, exposure=r.exposure,
                                 exit_rule=g.exit_rule, exit_buffer=g.exit_buffer,
                                 cooldown_s=g.cooldown_minutes * 60,
                                 catchup_margin=0.0 if sim is not None else 0.3,
                                 journal=self.journal)
        state.setdefault("cycles", 0)
        state.setdefault("resets_today", {})
        if state.get("mode", cfg.mode) != cfg.mode:
            raise RuntimeError(f"state dir belongs to mode {state['mode']!r}, config says {cfg.mode!r}")
        state["mode"] = cfg.mode
        self._stop = False

    # ------------------------------------------------------------------ helpers
    def equity(self) -> float:
        return float(self.ex.account()["totalMarginBalance"])

    def save(self):
        if self.sim is not None:
            self.state["sim"] = self.sim.to_dict()
        self.store.save(self.state)

    def flatten_all(self, reason: str):
        for s in self.cfg.symbols:
            res = StepResult(s)
            g = self.engine.grid(s)
            self.engine._cancel_grid_orders(s, g, res)
            pos = self.engine._position(s)
            if pos:
                self.engine._market(s, -pos, reason, res, f"flat{self.state['cycles']}-{s[:6]}")
            self.engine._save_grid(None, s)
            for e in res.errors:
                log.error("flatten %s: %s", s, e)
        self.journal("flatten_all", reason=reason)

    def halt(self, reason: str):
        log.critical("HALT: %s", reason)
        self.flatten_all(f"halt: {reason}")
        self.state.update(halted=True, halt_reason=reason)
        self.save()
        self.notifier.send(f"HALTED and flattened: {reason}. Review, then run `resume`.")

    # ------------------------------------------------------------------ one cycle
    def cycle(self) -> dict:
        st, cfg = self.state, self.cfg
        now = self.clock()
        if self.store.stop_requested():
            return {"status": "paused"}
        if st.get("halted"):
            return {"status": "halted", "reason": st.get("halt_reason")}
        prices = self.feed.update(now)
        equity = self.equity()

        day = now // DAY_MS
        if st.get("day") != day:                              # new UTC day
            st.update(day=day, day_start_equity=equity, resets_today={})
            if cfg.grid.regime_filter:
                st["regime_check"] = {s: self.strategy.regime_ok(s) for s in cfg.symbols}
            if st.get("cycles"):
                self.notifier.send(f"daily: equity {equity:,.2f} (peak {st.get('peak_equity', equity):,.2f}), "
                                   f"grids {', '.join(f'{s}:{d.get('bias')}' for s, d in st.get('grids', {}).items()) or 'none'}")
        st["peak_equity"] = max(float(st.get("peak_equity", equity)), equity)
        dd = 1 - equity / st["peak_equity"] if st["peak_equity"] > 0 else 0.0
        day_loss = 1 - equity / st["day_start_equity"] if st.get("day_start_equity") else 0.0
        if dd >= cfg.risk.max_drawdown:
            self.halt(f"drawdown {dd:.1%} from peak {st['peak_equity']:.2f}")
            return {"status": "halted"}
        if day_loss >= cfg.risk.max_daily_loss:
            self.halt(f"daily loss {day_loss:.1%}")
            return {"status": "halted"}

        st["cycles"] += 1
        out = {"status": "ok", "equity": equity, "steps": []}
        weights = cfg.weights()
        positions = {p["symbol"]: float(p["positionAmt"]) for p in self.ex.position_risk()}
        busy = {s for s in cfg.symbols if st["grids"].get(s) or positions.get(s)}
        for s in cfg.symbols:
            if s not in prices:
                continue
            if cfg.max_active and s not in busy and len(busy) >= cfg.max_active:
                continue                                   # rotation: all grid slots are taken
            g = self.engine.grid(s)
            if g and cfg.grid.recenter_hours and now - g.created_ms >= cfg.grid.recenter_hours * 3_600_000:
                res = StepResult(s)
                self.engine._cancel_grid_orders(s, g, res)
                self.engine._save_grid(None, s)                # next step re-creates it around the price
                self.journal("recenter", symbol=s, gid=g.gid)
            res = self.engine.step(s, prices[s], lambda s=s: self.strategy.plan(s),
                                   lambda s=s: self.equity() * weights[s], now, f"{st['cycles']}-{s[:8]}")
            out["steps"].append(res)
            if st["grids"].get(s):
                busy.add(s)
            elif (not res.market_trades and not positions.get(s)) or not self.engine._position(s):
                busy.discard(s)
            if res.action in ("reset", "stopped"):
                n = st["resets_today"].get(s, 0) + 1
                st["resets_today"][s] = n
                if n > cfg.risk.max_resets_per_day:
                    self.engine._cancel_grid_orders(s, self.engine.grid(s), res)
                    pos = self.engine._position(s)
                    if pos:
                        self.engine._market(s, -pos, "whipsaw guard", res, f"wg{st['cycles']}-{s[:8]}")
                    self.engine._save_grid(None, s)
                    st["cooldown_until"][s] = now + cfg.grid.cooldown_minutes * 60_000
                    self.notifier.send(f"{s}: {n} range exits today, cooling down {cfg.grid.cooldown_minutes} min")
            if res.action in ("created", "reset", "stopped"):
                g = self.engine.grid(s)
                self.notifier.send(f"{s}: grid {res.action}" + (f" centre {g.centre:.6g}, g {g.g:.3%}, n {g.n}, "
                                                                f"bias {g.bias}" if g else ""))
            for e in res.errors:
                log.warning("%s: %s", s, e)
                self.journal("error", symbol=s, error=e)
        self.save()
        return out

    # ------------------------------------------------------------------ loop
    def run_forever(self):
        def _sig(signum, _):
            log.warning("signal %s: stopping after this cycle", signum)
            self._stop = True
        for sgn in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sgn, _sig)
        log.info("grid bot started: mode=%s symbols=%s poll=%ss", self.cfg.mode, self.cfg.symbols, self.cfg.poll_seconds)
        errors = 0
        last_beat = 0
        while not self._stop:
            t0 = time.time()
            try:
                out = self.cycle()
                errors = 0
                if time.time() - last_beat > 600:
                    log.info("heartbeat: %s equity=%s", out.get("status"), f"{out.get('equity', float('nan')):.2f}")
                    last_beat = time.time()
            except Exception as e:                       # network blips etc.: back off, keep going
                errors += 1
                log.exception("cycle failed (%d in a row)", errors)
                self.journal("cycle_error", error=repr(e), streak=errors)
                if errors in (3, 20, 100):
                    self.notifier.send(f"cycle failing repeatedly ({errors}x): {e!r}")
                self.sleep(min(300, 5 * 2 ** min(errors, 6)))
                continue
            self.sleep(max(0.0, self.cfg.poll_seconds - (time.time() - t0)))
        self.save()
        log.info("grid bot stopped")
