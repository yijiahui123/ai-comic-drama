import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT

_DB_PATH = PROJECT_ROOT / "output" / "queue.db"

class QueueDB:
    """Lightweight SQLite-based persistent queue layer."""

    def __init__(self, db_path: Path = _DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        with conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS queue (
                    task_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    project_id TEXT,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')

    def add_task(self, task_id: str, kind: str, project_id: str | None, payload: dict[str, Any]) -> None:
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            conn.execute(
                "INSERT INTO queue (task_id, kind, project_id, payload, status, message, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (task_id, kind, project_id, json.dumps(payload), "queued", "Queued", now, now)
            )

    def get_pending_tasks(self) -> list[dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM queue WHERE status IN ('queued', 'running') ORDER BY created_at ASC")
        return [self._row_to_dict(row) for row in cursor.fetchall()]
        
    def get_task(self, task_id: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM queue WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def update_task_status(self, task_id: str, status: str, message: str) -> None:
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            conn.execute(
                "UPDATE queue SET status = ?, message = ?, updated_at = ? WHERE task_id = ?",
                (status, message, now, task_id)
            )

    def is_project_running(self, project_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT 1 FROM queue WHERE project_id = ? AND status IN ('queued', 'running') LIMIT 1",
            (project_id,)
        )
        return cursor.fetchone() is not None

    def delete_project_tasks(self, project_id: str) -> None:
        conn = self._get_conn()
        with conn:
            conn.execute("DELETE FROM queue WHERE project_id = ?", (project_id,))

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        if "payload" in d and isinstance(d["payload"], str):
            try:
                d["payload"] = json.loads(d["payload"])
            except json.JSONDecodeError:
                d["payload"] = {}
        return d

# Global instance
queue_db = QueueDB()
