"""Thin journal wrapper over bot.state.StateStore (shared state/lock/notifier implementation)."""
from __future__ import annotations


class Journal:
    def __init__(self, store):
        self.store = store

    def __call__(self, event: str, **data):
        self.store.journal(event, **data)
