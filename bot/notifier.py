"""
Minimal Telegram client (raw HTTP, only `requests`).

- send(text): push an alert to all authorized chat ids.
- poll(): long-poll getUpdates and yield (chat_id, text) for authorized chats only.
Authorization: only chat ids listed in cfg.tg_chat_ids may issue commands; if the list is
empty, the first chat that messages the bot is auto-learned (handy for first run).
"""
from __future__ import annotations
import requests


class Telegram:
    def __init__(self, token, chat_ids, timeout=25):
        self.token = token
        self.base = f"https://api.telegram.org/bot{token}"
        self.chat_ids = [str(c) for c in chat_ids]
        self.timeout = timeout
        self.offset = None
        self.s = requests.Session()

    @property
    def enabled(self):
        return bool(self.token)

    def send(self, text, chat_id=None):
        if not self.enabled:
            return
        targets = [chat_id] if chat_id else (self.chat_ids or [])
        for cid in targets:
            try:
                self.s.post(f"{self.base}/sendMessage", timeout=15,
                            data={"chat_id": cid, "text": text, "parse_mode": "HTML",
                                  "disable_web_page_preview": "true"})
            except requests.RequestException:
                pass

    def poll(self):
        """Yield (chat_id, text) for new messages from authorized chats."""
        if not self.enabled:
            return
        params = {"timeout": self.timeout}
        if self.offset is not None:
            params["offset"] = self.offset
        try:
            r = self.s.get(f"{self.base}/getUpdates", params=params, timeout=self.timeout + 10)
            data = r.json()
        except (requests.RequestException, ValueError):
            return
        if not data.get("ok"):
            return
        for upd in data["result"]:
            self.offset = upd["update_id"] + 1
            msg = upd.get("message") or upd.get("edited_message")
            if not msg or "text" not in msg:
                continue
            cid = str(msg["chat"]["id"])
            if not self.chat_ids:                 # auto-learn first chat on first run
                self.chat_ids = [cid]
            if cid not in self.chat_ids:
                continue
            yield cid, msg["text"].strip()
