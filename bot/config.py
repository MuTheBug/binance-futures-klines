"""
Configuration for the ECHO Engine live trading bot.

All settings come from environment variables (optionally loaded from a local `.env`
file next to this package). Nothing secret is hard-coded. SAFE DEFAULTS: the bot starts
in DRY_RUN mode (computes and reports orders but does NOT send them) and on TESTNET, so
you must consciously flip both to trade real money.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field


def _load_dotenv():
    """Minimal .env loader (no dependency on python-dotenv)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".env"), os.path.join(os.getcwd(), ".env")):
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _b(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _i(name, default):
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


@dataclass
class Config:
    # --- Binance ---
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True                  # SAFE DEFAULT: testnet
    recv_window: int = 5000

    # --- Telegram ---
    tg_token: str = ""
    tg_chat_ids: list = field(default_factory=list)   # authorized chat ids (whitelist)

    # --- Strategy / sizing ---
    leverage: float = 2.0                 # strategy leverage multiplier on the vol-targeted book
    target_vol: float = 0.40              # annualized vol target of the (gross~1) blend
    blend_trend: float = 0.50             # weight on TREND sleeve; rest on residual sleeve
    blend_echo: float = 0.80              # weight on ECHO book; rest on RIFT book
    max_gross: float = 3.0                # hard cap on total |notional|/equity (safety)
    binance_leverage: int = 5             # per-symbol leverage set on Binance (margin headroom)
    max_positions: int = 30               # keep only the strongest N target positions
                                          # (combined book holds ~27 names; truncating to 14 costs ~0.2 Sharpe)
    min_history_days: int = 60
    min_dollar_vol: float = 5_000_000.0   # trailing-30d median $vol eligibility floor
    klines_limit: int = 300               # daily bars fetched per symbol for the signal

    # --- Risk controls ---
    dry_run: bool = True                  # SAFE DEFAULT: do not send real orders
    max_drawdown_stop: float = 0.45       # circuit breaker: flatten + pause beyond this DD
    trail_dd: float = 0.25                # soft trailing throttle: halve book past this DD (0=off)
    trail_throttle: float = 0.5           # scale factor while throttled
    min_order_usdt: float = 5.0           # skip dust orders below this notional
    rebalance_band: float = 0.25          # only trade a name if |target-current| > band*|target_step|
    exec_style: str = "market"            # "maker" = post-only limit first, market fallback
    maker_wait_s: int = 45                # seconds to wait for a maker fill before falling back

    # --- Schedule ---
    rebalance_utc_hour: int = 0           # rebalance time (UTC), after the daily close
    rebalance_utc_minute: int = 5
    rebalance_every_days: int = 3         # cadence: 3d beats daily, phase-stable (see research/apex.py)
    summary_utc_hour: int = 8             # daily check-in summary alert time (UTC)
    poll_seconds: int = 10                # telegram long-poll / loop tick

    # --- Paths ---
    research_dir: str = ""
    state_path: str = ""

    @classmethod
    def load(cls) -> "Config":
        _load_dotenv()
        here = os.path.dirname(os.path.abspath(__file__))
        chat_ids = [c.strip() for c in os.environ.get("TG_CHAT_IDS", "").split(",") if c.strip()]
        return cls(
            api_key=os.environ.get("BINANCE_API_KEY", ""),
            api_secret=os.environ.get("BINANCE_API_SECRET", ""),
            testnet=_b("BINANCE_TESTNET", True),
            recv_window=_i("BINANCE_RECV_WINDOW", 5000),
            tg_token=os.environ.get("TG_TOKEN", ""),
            tg_chat_ids=chat_ids,
            leverage=_f("LEVERAGE", 2.0),
            target_vol=_f("TARGET_VOL", 0.40),
            blend_trend=_f("BLEND_TREND", 0.50),
            blend_echo=_f("BLEND_ECHO", 0.80),
            max_gross=_f("MAX_GROSS", 3.0),
            binance_leverage=_i("BINANCE_LEVERAGE", 5),
            max_positions=_i("MAX_POSITIONS", 30),
            min_history_days=_i("MIN_HISTORY_DAYS", 60),
            min_dollar_vol=_f("MIN_DOLLAR_VOL", 5_000_000.0),
            klines_limit=_i("KLINES_LIMIT", 300),
            dry_run=_b("DRY_RUN", True),
            max_drawdown_stop=_f("MAX_DRAWDOWN_STOP", 0.45),
            trail_dd=_f("TRAIL_DD", 0.25),
            trail_throttle=_f("TRAIL_THROTTLE", 0.5),
            min_order_usdt=_f("MIN_ORDER_USDT", 5.0),
            rebalance_band=_f("REBALANCE_BAND", 0.25),
            exec_style=os.environ.get("EXEC_STYLE", "market").strip().lower(),
            maker_wait_s=_i("MAKER_WAIT_S", 45),
            rebalance_utc_hour=_i("REBALANCE_UTC_HOUR", 0),
            rebalance_utc_minute=_i("REBALANCE_UTC_MINUTE", 5),
            rebalance_every_days=max(1, _i("REBALANCE_EVERY_DAYS", 3)),
            summary_utc_hour=_i("SUMMARY_UTC_HOUR", 8),
            poll_seconds=_i("POLL_SECONDS", 10),
            research_dir=os.environ.get("RESEARCH_DIR", os.path.join(os.path.dirname(here), "research")),
            state_path=os.environ.get("STATE_PATH", os.path.join(here, "state.json")),
        )

    def summary(self) -> str:
        net = "TESTNET" if self.testnet else "MAINNET"
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        return (f"net={net} mode={mode} lev={self.leverage:g}x targetVol={self.target_vol:g} "
                f"echo/rift={self.blend_echo:g}/{1-self.blend_echo:g} maxGross={self.max_gross:g}x "
                f"maxPos={self.max_positions} rebal={self.rebalance_utc_hour:02d}:"
                f"{self.rebalance_utc_minute:02d}UTC")
