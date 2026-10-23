"""Session persistence with per-session locking and timezone-aware timestamps.

Stored on disk as ``sessions/<uuid>.json`` shaped like::

    {"metadata": {"session_id": ..., "created_at": ..., "updated_at": ..., "title": ...},
     "messages": [{"role": ..., "content": ...}, ...]}
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import SessionContainer, SessionMeta
from .settings import Settings


def _now_iso() -> str:
    """UTC ISO-8601 timestamp, timezone-aware (replaces deprecated utcnow)."""
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dir: Path = settings.sessions_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        # A lock per session id so concurrent /chat requests to the same
        # session cannot corrupt the JSON file. Guarded by _locks_guard.
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------ #
    def _path(self, sid: str) -> Path:
        return self.dir / f"{sid}.json"

    def _lock_for(self, sid: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(sid)
            if lock is None:
                lock = threading.Lock()
                self._locks[sid] = lock
            return lock

    # ------------------------------------------------------------------ #
    def create(self, title: str = "Untitled Chat") -> tuple[str, list[dict[str, Any]]]:
        sid = str(uuid.uuid4())
        now = _now_iso()
        meta = SessionMeta(
            session_id=sid, created_at=now, updated_at=now, title=title
        )
        payload = SessionContainer(metadata=meta, messages=[])
        with self._lock_for(sid):
            with open(self._path(sid), "w", encoding="utf-8") as f:
                json.dump(payload.model_dump(), f, ensure_ascii=False, indent=2)
        return sid, payload.messages

    def load(self, sid: str) -> SessionContainer | None:
        p = self._path(sid)
        if not p.exists():
            return None
        with self._lock_for(sid):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        return _coerce_container(sid, data)

    def save(self, sid: str, messages: list[dict[str, Any]], title: str | None = None) -> None:
        existing = self.load(sid)
        now = _now_iso()
        if existing is None:
            meta = SessionMeta(
                session_id=sid, created_at=now, updated_at=now,
                title=title or "Untitled Chat",
            )
        else:
            meta = existing.metadata
            meta.updated_at = now
            if title is not None:
                meta.title = title
        payload = SessionContainer(metadata=meta, messages=messages)
        with self._lock_for(sid):
            with open(self._path(sid), "w", encoding="utf-8") as f:
                json.dump(payload.model_dump(), f, ensure_ascii=False, indent=2)

    def rename(self, sid: str, title: str) -> bool:
        existing = self.load(sid)
        if existing is None:
            return False
        existing.metadata.title = title
        existing.metadata.updated_at = _now_iso()
        with self._lock_for(sid):
            with open(self._path(sid), "w", encoding="utf-8") as f:
                json.dump(existing.model_dump(), f, ensure_ascii=False, indent=2)
        return True

    def delete(self, sid: str) -> bool:
        p = self._path(sid)
        if not p.exists():
            return False
        with self._lock_for(sid):
            p.unlink(missing_ok=True)
        return True

    def list_meta(self) -> list[SessionMeta]:
        out: list[SessionMeta] = []
        for fname in os.listdir(self.dir):
            if not fname.endswith(".json"):
                continue
            container = self.load(fname[:-5])
            if container is not None:
                out.append(container.metadata)
        # newest updated first
        out.sort(key=lambda m: m.updated_at, reverse=True)
        return out


def _coerce_container(sid: str, data: Any) -> SessionContainer:
    """Backward-compatible loader for legacy plain-message-list session files."""
    if isinstance(data, list):  # legacy: bare message list
        now = _now_iso()
        meta = SessionMeta(
            session_id=sid, created_at=now, updated_at=now, title="Untitled Chat"
        )
        return SessionContainer(metadata=meta, messages=data)
    return SessionContainer.model_validate(data)
