"""Outcome history H: a persistent PDE→architecture reuse cache + rollback helper.

The paper reuses successful (PDE, architecture) pairs across runs and rolls back to the
best-scoring iteration within a run. :class:`HistoryStore` persists cross-run outcomes
in SQLite; :func:`select_best_iteration` picks the winning snapshot inside one run.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


class HistoryStore:
    """SQLite-backed cache of per-run architecture outcomes."""

    def __init__(self, db_path: str | Path = "runs/history.db") -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    problem_key TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    rel_l2 REAL,
                    passed INTEGER NOT NULL,
                    ts REAL NOT NULL
                )
                """
            )

    def record(
        self,
        problem_key: str,
        architecture: str,
        quality_score: float,
        *,
        rel_l2: Optional[float] = None,
        passed: bool = False,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO outcomes"
                " (problem_key, architecture, quality_score, rel_l2, passed, ts)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (problem_key, architecture, quality_score, rel_l2, int(passed), time.time()),
            )

    def best_architecture(self, problem_key: str) -> Optional[dict[str, Any]]:
        """Return the best prior architecture for a problem, if any is cached."""

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT architecture, quality_score, rel_l2, passed FROM outcomes"
                " WHERE problem_key = ? ORDER BY quality_score DESC LIMIT 1",
                (problem_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "architecture": row[0],
            "quality_score": row[1],
            "rel_l2": row[2],
            "passed": bool(row[3]),
        }


def select_best_iteration(history: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Pick the highest-scoring iteration for accept/rollback (paper's S(Cᵗ) compare).

    Returns the history entry with the max ``quality_score`` (best-so-far), or None for
    an empty history. The graph uses this to roll back when a new attempt scored worse.
    """

    scored = [h for h in history if h.get("quality_score") is not None]
    if not scored:
        return None
    return max(scored, key=lambda h: h["quality_score"])
