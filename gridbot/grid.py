"""Grid geometry: levels, per-level quantities, inventory as a function of the level index.

Mirrors research/gridsim.py exactly (equal quote per level, geometric spacing):
    level k price   L_k = centre * (1+g)^k,  k = -n..n
    units bought at level k (and sold one level up): u_k ~= Q / L_k  (rounded to the lot step)
    inventory at index r:  inv(r) = sum_{k=r}^{n-1} u_k - c,
        c = 0 (long) | inv_raw(0) (neutral) | inv_raw(-n) (short)
The bot never trusts its own fill bookkeeping: each cycle it reads the real position and maps it
back to the nearest index r. That makes it robust to restarts, missed events and partial fills.
"""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

BIASES = ("long", "neutral", "short")


def _floor(x: float, step: Decimal) -> Decimal:
    return (Decimal(repr(float(x))) / step).to_integral_value(rounding=ROUND_FLOOR) * step


def _ceil(x: float, step: Decimal) -> Decimal:
    return (Decimal(repr(float(x))) / step).to_integral_value(rounding=ROUND_CEILING) * step


@dataclass
class Grid:
    symbol: str
    gid: str                  # short id used in clientOrderIds
    centre: float
    g: float                  # spacing
    n: int                    # levels per side
    q_quote: float            # quote per level
    bias: str
    step: str                 # lot step (string for JSON)
    tick: str
    min_notional: float
    created_ms: int = 0

    # ---------------------------------------------------------------- derived
    @property
    def _step(self) -> Decimal:
        return Decimal(self.step)

    @property
    def _tick(self) -> Decimal:
        return Decimal(self.tick)

    def level(self, k: int) -> float:
        return self.centre * (1 + self.g) ** k

    def buy_price(self, k: int) -> Decimal:
        return _floor(self.level(k), self._tick)

    def sell_price(self, k: int) -> Decimal:
        return _ceil(self.level(k), self._tick)

    def unit(self, k: int) -> Decimal:
        """Units bought at level k (sold at k+1). Never below the exchange minimum notional."""
        lv = self.level(k)
        u = _floor(self.q_quote / lv, self._step)
        need = _ceil(self.min_notional * 1.02 / lv, self._step)
        return max(u, need, self._step)

    def _raw(self, r: int) -> Decimal:
        return sum((self.unit(k) for k in range(r, self.n)), Decimal(0))

    def inv(self, r: int) -> Decimal:
        c = {"long": Decimal(0), "neutral": self._raw(0), "short": self._raw(-self.n)}[self.bias]
        return self._raw(r) - c

    def x(self, price: float) -> float:
        return math.log(price / self.centre) / math.log1p(self.g)

    # ---------------------------------------------------------------- state mapping
    def r_from_position(self, pos: float) -> tuple[int, float]:
        """Nearest level index for an actual position; residual in units."""
        best, res = 0, None
        for r in range(-self.n, self.n + 1):
            d = pos - float(self.inv(r))
            if res is None or abs(d) < abs(res):
                best, res = r, d
        return best, res

    def hysteresis(self, r: int, price: float, margin: float = 0.0) -> int:
        """Index the grid would be at if every resting order the price passed had filled.
        `margin` (in level units) requires the price to be that far beyond a level, so a resting
        order that is merely touched (and may still fill) is not duplicated by a catch-up trade."""
        x = self.x(price)
        down = max(min(r, math.ceil(x + margin - 1e-9)), -self.n)
        if down < r:
            return down
        return min(max(r, math.floor(x - margin + 1e-9)), self.n)

    def out_of_range(self, price: float, buffer: int) -> int:
        """-1 below the range, +1 above it, 0 inside (by more than `buffer` levels)."""
        x = self.x(price)
        return -1 if x < -self.n - buffer else (1 if x > self.n + buffer else 0)

    def ladder(self, r: int, depth: int) -> list[dict]:
        """Desired resting orders around index r: buys below, sells above (nearest `depth` each)."""
        out = []
        for j in range(1, depth + 1):
            k = r - j
            if k >= -self.n:
                out.append({"side": "BUY", "k": k, "price": self.buy_price(k), "qty": self.unit(k),
                            "cid": self.cid("B", k)})
            k = r + j
            if k <= self.n:
                out.append({"side": "SELL", "k": k, "price": self.sell_price(k), "qty": self.unit(k - 1),
                            "cid": self.cid("S", k)})
        return out

    def cid(self, side: str, k: int) -> str:
        return f"g{self.gid}-{side}{'m' if k < 0 else 'p'}{abs(k)}"

    def owns(self, client_order_id: str) -> bool:
        return client_order_id.startswith(f"g{self.gid}-")

    def max_notional(self) -> float:
        """Largest |inventory| notional the grid can reach (at the range edges)."""
        lo = abs(float(self.inv(-self.n))) * self.level(-self.n)
        hi = abs(float(self.inv(self.n))) * self.level(self.n)
        return max(lo, hi)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Grid":
        return cls(**d)


def q_for_exposure(bias: str, capital: float, g: float, n: int, exposure: float) -> float:
    """Quote per level so the largest |inventory notional| = exposure * capital (as in gridsim)."""
    a = 1 / (1 + g)
    k = {"long": (1 - a ** (2 * n)) / (1 - a),
         "neutral": (a ** -n - 1) / (1 - a),
         "short": (a ** (-2 * n) - 1) / (1 - a)}[bias]
    return exposure * capital / k
