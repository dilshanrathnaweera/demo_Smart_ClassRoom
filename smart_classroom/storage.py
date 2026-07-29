import sqlite3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import threading


DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "db.sqlite"


class Storage:
    """Lightweight SQLite storage for events, attendance, and occupancy history.

    Usage:
        st = Storage()  # uses default data/db.sqlite and creates folders
        st.add_event(level, message, confidences)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # allow multi-thread access; protect writes with a lock
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self):
        with self._conn:
            cur = self._conn.cursor()
            # events: raw inference events (timestamp UTC ISO, level, message, confidences JSON)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT,
                    confidences TEXT,
                    thumbnail_path TEXT
                )
                """
            )
            # attendance records: who/when/level/confidence
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person TEXT,
                    timestamp TEXT NOT NULL,
                    level TEXT,
                    confidence REAL,
                    thumbnail_path TEXT
                )
                """
            )
            # occupancy_history: periodic sampled confidences
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS occupancy_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    low REAL,
                    medium REAL,
                    high REAL
                )
                """
            )
            # ensure columns exist for older DBs
            cur.execute("PRAGMA table_info(events)")
            cols = [r[1] for r in cur.fetchall()]
            if 'thumbnail_path' not in cols:
                try:
                    cur.execute("ALTER TABLE events ADD COLUMN thumbnail_path TEXT")
                except Exception:
                    pass
            cur.execute("PRAGMA table_info(attendance_records)")
            cols2 = [r[1] for r in cur.fetchall()]
            if 'thumbnail_path' not in cols2:
                try:
                    cur.execute("ALTER TABLE attendance_records ADD COLUMN thumbnail_path TEXT")
                except Exception:
                    pass

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    # Events
    def add_event(self, level: str, message: str, confidences: Dict[str, float], timestamp: Optional[datetime] = None, thumbnail_path: Optional[str] = None) -> int:
        ts = (timestamp or datetime.utcnow()).isoformat()
        conf_json = json.dumps(confidences)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("INSERT INTO events (timestamp, level, message, confidences, thumbnail_path) VALUES (?, ?, ?, ?, ?)", (ts, level, message, conf_json, thumbnail_path))
            self._conn.commit()
            return cur.lastrowid

    def get_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM events ORDER BY timestamp DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        cur = self._conn.cursor()
        rows = cur.execute(q).fetchall()
        return [self._row_to_event_dict(r) for r in rows]

    def _row_to_event_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        conf = json.loads(row["confidences"]) if row["confidences"] else {}
        thumb = None
        try:
            thumb = row['thumbnail_path']
        except Exception:
            thumb = None
        return {"id": row["id"], "timestamp": row["timestamp"], "level": row["level"], "message": row["message"], "confidences": conf, "thumbnail_path": thumb}

    def export_events_csv(self, path: Path):
        import csv

        rows = self.get_events(limit=None)[::-1]  # oldest first
        with open(path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "timestamp", "level", "message", "low", "medium", "high", "thumbnail_path"])
            for r in rows:
                conf = r.get("confidences", {})
                writer.writerow([r["id"], r["timestamp"], r["level"], r["message"], conf.get("low"), conf.get("medium"), conf.get("high"), r.get('thumbnail_path')])

    # Attendance
    def add_attendance(self, person: str, timestamp: Optional[datetime] = None, level: Optional[str] = None, confidence: Optional[float] = None, thumbnail_path: Optional[str] = None) -> int:
        ts = (timestamp or datetime.utcnow()).isoformat()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("INSERT INTO attendance_records (person, timestamp, level, confidence, thumbnail_path) VALUES (?, ?, ?, ?, ?)", (person, ts, level, confidence, thumbnail_path))
            self._conn.commit()
            return cur.lastrowid

    def get_attendance(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM attendance_records ORDER BY timestamp DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        rows = self._conn.cursor().execute(q).fetchall()
        return [dict(r) for r in rows]

    def export_attendance_csv(self, path: Path):
        import csv

        rows = self.get_attendance(limit=None)[::-1]
        with open(path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "person", "timestamp", "level", "confidence", "thumbnail_path"])
            for r in rows:
                writer.writerow([r.get("id"), r.get("person"), r.get("timestamp"), r.get("level"), r.get("confidence"), r.get('thumbnail_path')])

    # Occupancy history
    def add_occupancy(self, low: float, medium: float, high: float, timestamp: Optional[datetime] = None) -> int:
        ts = (timestamp or datetime.utcnow()).isoformat()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("INSERT INTO occupancy_history (timestamp, low, medium, high) VALUES (?, ?, ?, ?)", (ts, low, medium, high))
            self._conn.commit()
            return cur.lastrowid

    def get_occupancy_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM occupancy_history ORDER BY timestamp DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        rows = self._conn.cursor().execute(q).fetchall()
        return [dict(r) for r in rows]


def _quick_demo():
    s = Storage()
    s.add_event('medium', 'test event', {'low': 0.1, 'medium': 0.8, 'high': 0.1})
    print('Events:', s.get_events(5))


if __name__ == '__main__':
    _quick_demo()
