"""Persistent bot state (JSON on disk): peak equity, equity history, runtime flags."""
from __future__ import annotations
import json
import os
import time


class State:
    def __init__(self, path):
        self.path = path
        self.data = {
            "paused": False,
            "peak_equity": 0.0,
            "last_rebalance_date": None,
            "equity_history": [],          # [[iso_ts, equity], ...]
            "tg_offset": None,
            "stopped_out": False,          # circuit breaker latched
        }
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.data.update(json.load(f))
            except (OSError, ValueError):
                pass
        return self

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    def record_equity(self, equity):
        self.data["peak_equity"] = max(self.data.get("peak_equity", 0.0) or 0.0, equity)
        hist = self.data.setdefault("equity_history", [])
        hist.append([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), round(equity, 4)])
        self.data["equity_history"] = hist[-2000:]
        self.save()

    def drawdown(self, equity):
        peak = self.data.get("peak_equity", 0.0) or 0.0
        return (equity / peak - 1.0) if peak > 0 else 0.0

    def __getitem__(self, k):
        return self.data.get(k)

    def __setitem__(self, k, v):
        self.data[k] = v
        self.save()
