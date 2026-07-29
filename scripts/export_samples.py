from smart_classroom.storage import Storage
from datetime import datetime, timedelta
from pathlib import Path
s = Storage()
# add sample events if none
if not s.get_events(limit=1):
    now = datetime.utcnow()
    s.add_event('low', 'sample low event', {'low':0.9,'medium':0.08,'high':0.02}, timestamp=now - timedelta(minutes=10))
    s.add_event('medium', 'sample medium event', {'low':0.1,'medium':0.8,'high':0.1}, timestamp=now - timedelta(minutes=5))
    s.add_event('high', 'sample high event', {'low':0.01,'medium':0.09,'high':0.9}, timestamp=now)
# add attendance
if not s.get_attendance(limit=1):
    s.add_attendance('Alice', timestamp=datetime.utcnow(), level='medium', confidence=0.78)
    s.add_attendance('Bob', timestamp=datetime.utcnow(), level='high', confidence=0.92)
# export
outdir = Path('data/exports')
outdir.mkdir(parents=True, exist_ok=True)
sev = outdir / 'events_export.csv'
s.export_events_csv(sev)
sa = outdir / 'attendance_export.csv'
s.export_attendance_csv(sa)
print('Wrote', sev)
print('Wrote', sa)
