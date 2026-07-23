"""SQLite persistence for locally generated recommendation sessions."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .models import Recommendation, UserProfile


@dataclass(frozen=True)
class SavedSession:
    id: str
    title: str
    profile: dict[str, Any]
    recommendations: list[dict[str, Any]]
    created_at: str


class RecommendationStore:
    """Persist history without a server or cloud account."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Commit or roll back a transaction and always release the file."""

        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    recommendations_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save(
        self,
        profile: UserProfile,
        recommendations: tuple[Recommendation, ...],
    ) -> str:
        session_id = uuid.uuid4().hex
        title = recommendations[0].role.name if recommendations else "Career exploration"
        created_at = datetime.now(UTC).isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO recommendation_sessions
                    (id, title, profile_json, recommendations_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    title,
                    json.dumps(profile.to_dict(), ensure_ascii=False),
                    json.dumps(
                        [item.to_dict() for item in recommendations],
                        ensure_ascii=False,
                    ),
                    created_at,
                ),
            )
        return session_id

    def list_recent(self, *, limit: int = 8) -> tuple[SavedSession, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, title, profile_json, recommendations_json, created_at
                FROM recommendation_sessions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(self._deserialize(row) for row in rows)

    def get(self, session_id: str) -> SavedSession | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, title, profile_json, recommendations_json, created_at
                FROM recommendation_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        return self._deserialize(row) if row else None

    @staticmethod
    def _deserialize(row: sqlite3.Row) -> SavedSession:
        return SavedSession(
            id=row["id"],
            title=row["title"],
            profile=json.loads(row["profile_json"]),
            recommendations=json.loads(row["recommendations_json"]),
            created_at=row["created_at"],
        )
