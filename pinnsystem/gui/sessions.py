"""Persistent session index for the GUI.

Each "Run" is a session with its own ``thread_id`` (== workspace run id), so the
LangGraph SQLite checkpointer keeps its full state and a follow-up can resume it. This
store persists the lightweight *metadata + transcript* next to the checkpoints so the
sidebar can list past sessions and reopening one shows its history. Graph state itself
lives in the checkpointer, not here.

The store is framework-free (stdlib + json) so it is unit-testable without nicegui.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Session:
    """One GUI session's metadata and its rendered transcript entries."""

    session_id: str
    title: str
    created_at: str
    workspace_dir: str
    transcript: list[dict] = field(default_factory=list)


class SessionStore:
    """JSON-backed list of :class:`Session`, newest first."""

    def __init__(self, runs_dir: str | Path = "runs") -> None:
        self.runs_dir = Path(runs_dir)
        self.path = self.runs_dir / "sessions.json"
        self._sessions: dict[str, Session] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for item in raw.get("sessions", []):
            try:
                sess = Session(**item)
            except TypeError:  # skip malformed/legacy rows rather than crash the GUI
                continue
            self._sessions[sess.session_id] = sess

    def _save(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        payload = {"sessions": [asdict(s) for s in self.list()]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list(self) -> list[Session]:
        """All sessions, newest first."""

        return sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def create(self, session_id: str, workspace_dir: str, title: str = "") -> Session:
        sess = Session(
            session_id=session_id,
            title=title or "Untitled run",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            workspace_dir=workspace_dir,
        )
        self._sessions[session_id] = sess
        self._save()
        return sess

    def set_title(self, session_id: str, title: str) -> None:
        sess = self._sessions.get(session_id)
        if sess and title:
            sess.title = title[:80]
            self._save()

    def append_entry(self, session_id: str, entry: dict[str, Any]) -> None:
        """Persist one transcript entry so reopening the session shows it."""

        sess = self._sessions.get(session_id)
        if sess is None:
            return
        sess.transcript.append(entry)
        self._save()
