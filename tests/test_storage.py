import tempfile
from pathlib import Path
import os
import csv

from smart_classroom.storage import Storage


def test_storage_creates_db_and_tables(tmp_path):
    db_file = tmp_path / "test_db.sqlite"
    s = Storage(db_file)
    # ensure file created
    assert db_file.exists()
    # check tables exist by querying sqlite_master
    cur = s._conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    names = {r[0] for r in cur.fetchall()}
    assert 'events' in names
    assert 'attendance_records' in names
    assert 'occupancy_history' in names
    s.close()


def test_add_get_export_events(tmp_path):
    db_file = tmp_path / "test_db2.sqlite"
    s = Storage(db_file)
    eid = s.add_event('high', 'crowded', {'low': 0.01, 'medium': 0.09, 'high': 0.9})
    assert isinstance(eid, int)
    events = s.get_events()
    assert len(events) >= 1
    # export
    out = tmp_path / 'events.csv'
    s.export_events_csv(out)
    assert out.exists()
    with open(out, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert rows[0] == ['id', 'timestamp', 'level', 'message', 'low', 'medium', 'high', 'thumbnail_path']
    s.close()


def test_attendance_crud_and_export(tmp_path):
    db_file = tmp_path / "test_db3.sqlite"
    s = Storage(db_file)
    aid = s.add_attendance('Alice', level='medium', confidence=0.8)
    assert isinstance(aid, int)
    att = s.get_attendance()
    assert len(att) >= 1
    out = tmp_path / 'attendance.csv'
    s.export_attendance_csv(out)
    assert out.exists()
    with open(out, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert rows[0] == ['id', 'person', 'timestamp', 'level', 'confidence', 'thumbnail_path']
    s.close()


def test_occupancy_history(tmp_path):
    db_file = tmp_path / "test_db4.sqlite"
    s = Storage(db_file)
    oid = s.add_occupancy(0.1, 0.2, 0.7)
    assert isinstance(oid, int)
    hist = s.get_occupancy_history()
    assert len(hist) >= 1
    s.close()
