"""Crash-safe JSON state, an append-only JSONL journal, a single-instance lock, and notifications."""
from __future__ import annotations
import fcntl
import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import requests

log = logging.getLogger(__name__)


class StateStore:
    def __init__(self, state_dir: str):
        self.dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.path = os.path.join(state_dir, "state.json")
        self.journal_path = os.path.join(state_dir, "journal.jsonl")
        self.equity_path = os.path.join(state_dir, "equity.csv")
        self.stop_path = os.path.join(state_dir, "STOP")

    def load(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        with open(self.path) as f:
            return json.load(f)

    def save(self, state: dict):
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".state.", suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def journal(self, event: str, **data):
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def record_equity(self, day: str, equity: float, gross: float, n_pos: int):
        new = not os.path.exists(self.equity_path)
        with open(self.equity_path, "a") as f:
            if new:
                f.write("day,equity,gross_notional,positions\n")
            f.write(f"{day},{equity:.4f},{gross:.4f},{n_pos}\n")

    def stop_requested(self) -> bool:
        return os.path.exists(self.stop_path)


class InstanceLock:
    """Prevents two bot processes from trading the same state directory."""

    def __init__(self, state_dir: str):
        os.makedirs(state_dir, exist_ok=True)
        self.path = os.path.join(state_dir, "bot.lock")
        self.fh = None

    def __enter__(self):
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another bot instance holds {self.path}") from None
        self.fh.write(str(os.getpid()))
        self.fh.flush()
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


class Notifier:
    def __init__(self, webhook_url: str = "", telegram_token: str = "", telegram_chat_id: str = "",
                 prefix: str = "[v4 bot]"):
        self.webhook_url, self.tg_token, self.tg_chat = webhook_url, telegram_token, telegram_chat_id
        self.prefix = prefix

    def send(self, text: str):
        text = f"{self.prefix} {text}"
        log.info("notify: %s", text)
        try:
            if self.webhook_url:
                requests.post(self.webhook_url, json={"text": text, "content": text}, timeout=10)
            if self.tg_token and self.tg_chat:
                requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                              json={"chat_id": self.tg_chat, "text": text}, timeout=10)
        except Exception as e:                     # notifications must never break trading
            log.warning("notification failed: %s", e)
