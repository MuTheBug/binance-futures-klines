"""Bot configuration: TOML file + environment variables for secrets.

Secrets (API keys) are NEVER read from the config file; they come from the environment:
    BINANCE_API_KEY, BINANCE_API_SECRET
"""
from __future__ import annotations
import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass

# Tokenized stocks / metals / FX perps: excluded so the edge stays pure crypto (matches research).
DEFAULT_EXCLUDE = [
    "NVDA", "TSLA", "MSTR", "SPY", "QQQ", "INTC", "MU", "MRVL", "AMD", "SNDK", "SKHYNIX", "NOK",
    "EWY", "SOXL", "CRCL", "SPCX", "XAG", "XAU", "XAUT", "PAXG", "CL", "BZ", "SLX", "BILL", "COIN",
    "GENIUS", "CHIP",
]

MAX_ALLOWED_LEVERAGE = 5.0   # research: growth turns negative and ruin appears beyond ~6x


@dataclass
class StrategyCfg:
    w_trend: float = 0.70             # v3 trend book
    w_pump: float = 0.30              # pump-follow sleeve
    target_vol: float = 0.40          # per-sleeve annualized vol target
    max_sleeve_leverage: float = 3.0  # cap on each sleeve's vol-target scale
    pump_threshold: float = 0.25      # coin minus BTC daily return that triggers the sleeve
    pump_hold_days: int = 5
    min_history: int = 60             # days of history before a coin is eligible
    min_dollar_volume: float = 5e6    # trailing-30d median daily $ volume
    history_bars: int = 400           # daily bars fetched per symbol (>= 250 recommended)
    hedge_symbol: str = "BTCUSDT"
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))


@dataclass
class RiskCfg:
    leverage: float = 1.5             # multiplies strategy weights (research: stay <= 2)
    max_gross: float = 3.0            # hard cap on sum |notional| / equity
    max_position: float = 0.50        # hard cap on one symbol's |notional| / equity
    max_drawdown: float = 0.35        # flatten + halt when equity falls this far below its peak
    max_cycle_loss: float = 0.20      # flatten + halt if equity falls this much between cycles
    rebalance_band: float = 0.01      # skip trades smaller than this fraction of equity
    stop_loss_pct: float = 0.0        # optional exchange-side catastrophe stop per position; 0 = off


@dataclass
class ExchangeCfg:
    testnet: bool = False
    base_url: str = ""                # override; default mainnet / testnet URL
    margin_type: str = "CROSSED"      # CROSSED or ISOLATED
    exchange_leverage: int = 5        # per-symbol leverage setting on Binance (margin headroom only)
    recv_window: int = 5000
    min_quote_volume_24h: float = 1e6 # cheap prefilter before fetching klines
    max_workers: int = 8
    adopt_existing_positions: bool = False


@dataclass
class PaperCfg:
    initial_equity: float = 10_000.0
    fee: float = 0.0005               # taker fee per side
    slippage: float = 0.0002
    data_source: str = "binance"      # binance (live public data) | csv (offline repo data)
    csv_dir: str = "."


@dataclass
class ScheduleCfg:
    run_minute_utc: int = 5           # minutes after the 00:00 UTC daily close
    retry_minutes: int = 5
    max_retries: int = 6


@dataclass
class NotifyCfg:
    webhook_url: str = ""             # optional: POST {"text": ...} (Slack/Discord-compatible)
    telegram_token: str = ""
    telegram_chat_id: str = ""


@dataclass
class Config:
    mode: str = "paper"               # paper | live
    state_dir: str = "bot_state"
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    exchange: ExchangeCfg = field(default_factory=ExchangeCfg)
    paper: PaperCfg = field(default_factory=PaperCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    notify: NotifyCfg = field(default_factory=NotifyCfg)
    api_key: str = field(default="", repr=False)
    api_secret: str = field(default="", repr=False)

    def validate(self) -> "Config":
        s, r = self.strategy, self.risk
        if self.mode not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got {self.mode!r}")
        if s.w_trend < 0 or s.w_pump < 0 or s.w_trend + s.w_pump <= 0:
            raise ValueError("strategy weights must be >= 0 and not both zero")
        if not 0 < r.leverage <= MAX_ALLOWED_LEVERAGE:
            raise ValueError(f"risk.leverage must be in (0, {MAX_ALLOWED_LEVERAGE}]")
        if not 0 < r.max_gross <= 10:
            raise ValueError("risk.max_gross must be in (0, 10]")
        if not 0 < r.max_drawdown < 1 or not 0 < r.max_cycle_loss < 1:
            raise ValueError("risk.max_drawdown and risk.max_cycle_loss must be in (0, 1)")
        if not 0 <= r.stop_loss_pct < 1:
            raise ValueError("risk.stop_loss_pct must be in [0, 1)")
        if s.history_bars < 200:
            raise ValueError("strategy.history_bars must be >= 200 (indicator warm-up)")
        if self.exchange.margin_type not in ("CROSSED", "ISOLATED"):
            raise ValueError("exchange.margin_type must be CROSSED or ISOLATED")
        if self.paper.data_source not in ("binance", "csv"):
            raise ValueError("paper.data_source must be binance or csv")
        if not 0 <= self.schedule.run_minute_utc < 24 * 60:
            raise ValueError("schedule.run_minute_utc must be in [0, 1440)")
        if self.mode == "live" and (not self.api_key or not self.api_secret):
            raise ValueError("live mode needs BINANCE_API_KEY and BINANCE_API_SECRET in the environment")
        return self


def _fill(dc, data: dict, where: str):
    names = {f.name: f for f in fields(dc)}
    for k, v in data.items():
        if k not in names:
            raise ValueError(f"unknown config key {where}{k!r}")
        cur = getattr(dc, k)
        if is_dataclass(cur):
            if not isinstance(v, dict):
                raise ValueError(f"config section {where}{k} must be a table")
            _fill(cur, v, f"{where}{k}.")
        else:
            if isinstance(cur, float) and isinstance(v, int):
                v = float(v)
            if cur is not None and not isinstance(v, type(cur)):
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
