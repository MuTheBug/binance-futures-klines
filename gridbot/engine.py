"""Grid engine: one reconcile step per symbol, identical for paper, replay and live.

Each step:
  1. read the real position and the bot's resting orders for the symbol
  2. no grid yet -> create one centred on the current price (if the strategy allows), and trade at
     market to the grid's starting inventory
  3. price beyond the range by `exit_buffer` levels -> reset (re-centre) or stop (flatten + cool down)
  4. map the position to the nearest level index r; if the price has passed levels whose orders
     did not fill (gap, downtime, rejected post-only), catch up at market to the index the grid
     should be at (hysteresis rule, same as the research simulator)
  5. make the resting ladder exactly: post-only buys below r, sells above r
The exchange is anything with the BinanceFuturesClient method names (live client or SimExchange).
"""
from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from decimal import Decimal

from bot.binance import AmbiguousOrderError, BinanceAPIError, SymbolFilters
from .grid import Grid, q_for_exposure, _floor

log = logging.getLogger(__name__)


@dataclass
class GridPlan:
    """What the strategy layer decides when a grid (re)starts."""
    allowed: bool
    g: float = 0.0
    bias: str = "neutral"
    reason: str = ""


@dataclass
class StepResult:
    symbol: str
    action: str = "ok"                 # ok | created | reset | stopped | flat | halted
    placed: int = 0
    cancelled: int = 0
    market_trades: list = field(default_factory=list)
    errors: list = field(default_factory=list)


class GridEngine:
    def __init__(self, exchange, filters: dict[str, SymbolFilters], state: dict, *, n: int,
                 exposure: float, exit_rule: str = "reset", exit_buffer: int = 1, cooldown_s: int = 0,
                 depth: int | None = None, catchup_margin: float = 0.0, journal=None):
        self.ex, self.filters, self.state = exchange, filters, state
        self.n, self.exposure = n, exposure
        self.exit_rule, self.exit_buffer, self.cooldown_s = exit_rule, exit_buffer, cooldown_s
        self.depth = depth                    # None = every level in the range (as in research)
        self.catchup_margin = catchup_margin  # live: 0.3 so touched-but-unfilled orders aren't doubled
        self.journal = journal or (lambda event, **kw: None)
        state.setdefault("grids", {})
        state.setdefault("cooldown_until", {})

    # ---------------------------------------------------------------- helpers
    def grid(self, sym) -> Grid | None:
        d = self.state["grids"].get(sym)
        return Grid.from_dict(d) if d else None

    def _save_grid(self, g: Grid | None, sym: str):
        if g is None:
            self.state["grids"].pop(sym, None)
        else:
            self.state["grids"][sym] = g.to_dict()

    def _market(self, sym, dq: float, reason: str, res: StepResult, tag: str):
        """Market order for signed dq (base units), rounded to step; reduce-only when shrinking."""
        f = self.filters[sym]
        q = _floor(abs(dq), f.step)
        if q <= 0:
            return
        side = "BUY" if dq > 0 else "SELL"
        pos = self._position(sym)
        reduce = pos != 0 and (pos > 0) != (dq > 0) and abs(dq) <= abs(pos) + 1e-12
        params = dict(symbol=sym, side=side, type="MARKET", quantity=format(q.normalize(), "f"),
                      newClientOrderId=f"gm-{tag}"[:36], newOrderRespType="RESULT")
        if reduce:
            params["reduceOnly"] = "true"
        try:
            r = self.ex.new_order(**params)
        except AmbiguousOrderError as e:
            res.errors.append(f"{sym} market {side} {q}: ambiguous ({e}); will reconcile next step")
            return
        except BinanceAPIError as e:
            res.errors.append(f"{sym} market {side} {q}: {e}")
            return
        trade = {"symbol": sym, "side": side, "qty": float(r.get("executedQty", q)),
                 "price": float(r.get("avgPrice", 0) or 0), "reason": reason}
        res.market_trades.append(trade)
        self.journal("market", **trade)

    def _position(self, sym) -> float:
        for p in self.ex.position_risk():
            if p["symbol"] == sym:
                return float(p["positionAmt"])
        return 0.0

    def _cancel_grid_orders(self, sym, g: Grid | None, res: StepResult):
        for o in self.ex.open_orders(sym):
            cid = o.get("clientOrderId", "")
            if g is None or g.owns(cid) or cid.startswith("g"):
                try:
                    self.ex.cancel_order(sym, cid)
                    res.cancelled += 1
                except BinanceAPIError as e:
                    if e.code != -2011:                 # already gone
                        res.errors.append(f"cancel {cid}: {e}")

    def new_grid(self, sym, price: float, plan: GridPlan, capital: float, now_ms: int) -> Grid | None:
        """Largest grid (<= self.n levels a side) whose orders meet the exchange minimum AND whose
        lot-size rounding does not push the real maximum inventory more than 25% over plan."""
        f = self.filters[sym]
        for n in range(self.n, 0, -1):              # small account: fewer, larger levels
            q = q_for_exposure(plan.bias, capital, plan.g, n, self.exposure)
            if q < f.min_notional * 1.02:
                continue
            g = Grid(symbol=sym, gid=uuid.uuid4().hex[:6], centre=price, g=plan.g, n=n, q_quote=q,
                     bias=plan.bias, step=str(f.step), tick=str(f.tick), min_notional=f.min_notional,
                     created_ms=now_ms)
            if g.max_notional() <= self.exposure * capital * 1.25:
                return g
        return None

    # ---------------------------------------------------------------- the step
    def step(self, sym: str, price: float, plan_fn, capital, now_ms: int, tag: str,
             check_exit: bool = True, allow_create: bool = True) -> StepResult:
        """plan_fn() -> GridPlan is called only when a grid must be (re)created; `capital` may be a
        number or a zero-argument callable (evaluated only when a grid is sized). check_exit=False
        skips range-exit/regime handling (replay checks them only at bar closes, like research)."""
        cap = capital if callable(capital) else (lambda: capital)
        res = StepResult(sym)
        g = self.grid(sym)
        pos = self._position(sym)

        if g is None:
            if not allow_create:
                res.action = "idle"
                return res
            if now_ms < self.state["cooldown_until"].get(sym, 0):
                res.action = "cooldown"
                return res
            plan = plan_fn()
            if not plan.allowed:
                if pos:
                    self._cancel_grid_orders(sym, None, res)
                    self._market(sym, -pos, f"flat: {plan.reason}", res, f"{tag}-f")
                res.action = "flat"
                return res
            g = self.new_grid(sym, price, plan, cap(), now_ms)
            if g is None:
                res.action = "flat"
                res.errors.append(f"{sym}: capital {cap():.2f} too small for one grid level")
                return res
            self._cancel_grid_orders(sym, None, res)
            self._market(sym, float(g.inv(0)) - pos, "grid start", res, f"{tag}-s")
            self._save_grid(g, sym)
            self.journal("grid_start", **g.to_dict())
            res.action = "created"
            pos = self._position(sym)
        else:
            out = g.out_of_range(price, self.exit_buffer) if check_exit and self.exit_rule != "hold" else 0
            plan = None
            if check_exit and out == 0 and self.state.get("regime_check", {}).get(sym) is False:
                plan = GridPlan(False, reason="regime off")
            if out or plan is not None:
                self._cancel_grid_orders(sym, g, res)
                if self.exit_rule == "reset" and plan is None:
                    plan = plan_fn()
                    if plan.allowed:
                        ng = self.new_grid(sym, price, plan, cap(), now_ms)
                        if ng is not None:
                            self._market(sym, float(ng.inv(0)) - pos, "reset", res, f"{tag}-r")
                            self._save_grid(ng, sym)
                            self.journal("grid_reset", old=g.gid, side=out, **ng.to_dict())
                            res.action = "reset"
                            g, pos = ng, self._position(sym)
                        else:
                            plan = GridPlan(False, reason="capital too small")
                if res.action != "reset":
                    self._market(sym, -pos, "stop" if out else plan.reason, res, f"{tag}-x")
                    self._save_grid(None, sym)
                    if out and self.cooldown_s:
                        self.state["cooldown_until"][sym] = now_ms + self.cooldown_s * 1000
                    self.journal("grid_stop", gid=g.gid, side=out, reason=(plan.reason if plan else "range exit"))
                    res.action = "stopped"
                    return res

        # catch up on levels the price passed without fills
        r, resid = g.r_from_position(pos)
        r_h = g.hysteresis(r, price, self.catchup_margin)
        if r_h != r:
            self._market(sym, float(g.inv(r_h)) - pos, f"catch-up {r}->{r_h}", res, f"{tag}-c")
            pos = self._position(sym)
            r, resid = g.r_from_position(pos)

        # reconcile the resting ladder
        want = {o["cid"]: o for o in g.ladder(r, self.depth or 2 * g.n)}
        have = {o["clientOrderId"]: o for o in self.ex.open_orders(sym) if g.owns(o.get("clientOrderId", ""))}
        for cid, o in list(have.items()):
            w = want.get(cid)
            partial = float(o.get("executedQty", 0) or 0) > 0
            if w is None and partial:
                continue                                 # let a partially filled order finish
            if w is None or Decimal(o["price"]) != w["price"] or Decimal(o["origQty"]) != w["qty"]:
                try:
                    self.ex.cancel_order(sym, cid)
                    res.cancelled += 1
                except BinanceAPIError as e:
                    if e.code != -2011:
                        res.errors.append(f"cancel {cid}: {e}")
                    continue
                have[cid] = None
        for cid, w in want.items():
            if have.get(cid) is not None:
                continue
            try:
                self.ex.new_order(symbol=sym, side=w["side"], type="LIMIT", timeInForce="GTX",
                                  quantity=format(w["qty"].normalize(), "f"),
                                  price=format(w["price"].normalize(), "f"), newClientOrderId=cid)
                res.placed += 1
            except BinanceAPIError as e:
                if e.code == -5022:                      # would cross: price moved, catch up next step
                    continue
                res.errors.append(f"place {cid}: {e}")
                if e.code == -2019:                      # margin: stop adding opening orders
                    break
            except AmbiguousOrderError as e:
                res.errors.append(f"place {cid}: ambiguous ({e}); reconciled next step")
        return res
