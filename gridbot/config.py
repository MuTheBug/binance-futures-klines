"""Grid bot configuration: TOML file + environment variables for secrets.

API keys are never read from the file: export BINANCE_API_KEY / BINANCE_API_SECRET.
"""
from __future__ import annotations
import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass

MAX_EXPOSURE = 3.0


@dataclass
class GridCfg:
    levels: int = 10                  # levels on each side of the centre (reduced automatically if
                                      # the account is too small for the exchange minimum order)
    spacing: str = "vol"              # "vol": g = spacing_k * sigma_1h(7d) | "fixed": g = spacing_fixed
    spacing_k: float = 6.0
    spacing_fixed: float = 0.02
    spacing_min: float = 0.004        # floor on g (round trip must beat fees by a wide margin)
    spacing_max: float = 0.10
    bias: str = "neutral"             # neutral | long | short | trend
    trend_threshold: float = 0.3      # |v3 daily trend strength| above this = trending
    regime_filter: bool = True        # only run grids while |trend| < trend_threshold
    exit_rule: str = "stop"           # stop (flatten + cooldown) | reset (re-centre) | hold
    exit_buffer: int = 1              # levels beyond the range before the exit rule fires
    cooldown_minutes: int = 1440
    recenter_hours: int = 0           # periodic re-centre (0 = only on range exits)


@dataclass
class RiskCfg:
    exposure: float = 1.0             # max |inventory notional| / capital allocated to a symbol
    max_drawdown: float = 0.30        # flatten + halt when equity falls this far below its peak
    max_daily_loss: float = 0.15      # flatten + halt on this loss since 00:00 UTC
    max_resets_per_day: int = 8       # per symbol; beyond it the symbol cools down (whipsaw guard)
    exchange_leverage: int = 5        # Binance per-symbol leverage setting (margin headroom only)
    margin_type: str = "CROSSED"


@dataclass
class ExchangeCfg:
    testnet: bool = False
    base_url: str = ""
    recv_window: int = 5000


@dataclass
class PaperCfg:
    initial_balance: float = 100.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    slippage: float = 0.0005
    through: float = 0.0              # extra trade-through needed for a paper limit fill
    csv_dir: str = "."                # hourly CSVs for `replay` ("." = repo root)


@dataclass
class NotifyCfg:
    webhook_url: str = ""
    telegram_token: str = ""
    telegram_chat_id: str = ""


@dataclass
class Config:
    mode: str = "paper"               # paper | live
    state_dir: str = "gridbot_state"
    symbols: list[str] = field(default_factory=lambda: ["DOGEUSDT"])
    allocation: list[float] = field(default_factory=list)   # fractions of equity; default equal
    max_active: int = 0               # rotation: run grids on at most this many of `symbols` at once,
                                      # each sized equity / max_active (0 = every symbol, `allocation`)
    poll_seconds: int = 10
    grid: GridCfg = field(default_factory=GridCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    exchange: ExchangeCfg = field(default_factory=ExchangeCfg)
    paper: PaperCfg = field(default_factory=PaperCfg)
    notify: NotifyCfg = field(default_factory=NotifyCfg)
    api_key: str = field(default="", repr=False)
    api_secret: str = field(default="", repr=False)

    def weights(self) -> dict[str, float]:
        if self.max_active:
            return {s: 1.0 / self.max_active for s in self.symbols}
        if self.allocation:
            return dict(zip(self.symbols, self.allocation))
        return {s: 1.0 / len(self.symbols) for s in self.symbols}

    def validate(self) -> "Config":
        g, r = self.grid, self.risk
        if self.mode not in ("paper", "live"):
            raise ValueError("mode must be 'paper' or 'live'")
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be a non-empty list without duplicates")
        if any(not s.endswith("USDT") for s in self.symbols):
            raise ValueError("symbols must be USDT-margined perpetuals, e.g. DOGEUSDT")
        if self.allocation and (len(self.allocation) != len(self.symbols) or abs(sum(self.allocation) - 1) > 1e-6
                                or min(self.allocation) <= 0):
            raise ValueError("allocation must have one positive fraction per symbol, summing to 1")
        if not 0 <= self.max_active <= len(self.symbols):
            raise ValueError("max_active must be between 0 and the number of symbols")
        if self.max_active and self.allocation:
            raise ValueError("use either max_active (rotation) or allocation, not both")
        if not 1 <= g.levels <= 50:
            raise ValueError("grid.levels must be 1..50")
        if g.spacing not in ("vol", "fixed") or g.bias not in ("neutral", "long", "short", "trend"):
            raise ValueError("grid.spacing must be vol|fixed and grid.bias neutral|long|short|trend")
        if g.exit_rule not in ("reset", "stop", "hold"):
            raise ValueError("grid.exit_rule must be reset|stop|hold")
        if not 0 < g.spacing_min <= g.spacing_max < 0.5 or not 0.001 <= g.spacing_fixed < 0.5:
            raise ValueError("grid spacing bounds out of range")
        if not 0 < r.exposure <= MAX_EXPOSURE:
            raise ValueError(f"risk.exposure must be in (0, {MAX_EXPOSURE}]")
        if not 0 < r.max_drawdown < 1 or not 0 < r.max_daily_loss < 1:
            raise ValueError("risk.max_drawdown / max_daily_loss must be in (0, 1)")
        if r.margin_type not in ("CROSSED", "ISOLATED"):
            raise ValueError("risk.margin_type must be CROSSED or ISOLATED")
        if self.poll_seconds < 2:
            raise ValueError("poll_seconds must be >= 2")
        if self.mode == "live" and (not self.api_key or not self.api_secret):
            raise ValueError("live mode needs BINANCE_API_KEY and BINANCE_API_SECRET in the environment")
        return self


def _fill(dc, data: dict, where: str):
    names = {f.name for f in fields(dc)}
    for k, v in data.items():
        if k not in names:
            raise ValueError(f"unknown config key {where}{k!r}")
        cur = getattr(dc, k)
        if is_dataclass(cur):
            if not isinstance(v, dict):
                raise ValueError(f"config section {where}{k} must be a table")
            _fill(cur, v, f"{where}{k}.")
            continue
        if isinstance(cur, float) and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        if isinstance(cur, list):
            if not isinstance(v, list):
                raise ValueError(f"config {where}{k} must be a list")
            v = [float(x) if isinstance(x, int) and k == "allocation" else x for x in v]
        elif not isinstance(v, type(cur)):
            raise ValueError(f"config {where}{k} must be {type(cur).__name__}, got {type(v).__name__}")
        setattr(dc, k, v)


def load_config(path: str | None = None, env: dict | None = None) -> Config:
    env = os.environ if env is None else env
    cfg = Config()
    if path:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        for secret in ("api_key", "api_secret"):
            if secret in raw:
                raise ValueError(f"do not put {secret} in the config file; use environment variables")
        _fill(cfg, raw, "")
    cfg.api_key = env.get("BINANCE_API_KEY", "")
    cfg.api_secret = env.get("BINANCE_API_SECRET", "")
    return cfg.validate()
