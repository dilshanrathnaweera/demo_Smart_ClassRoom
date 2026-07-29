import numpy as np
from pathlib import Path
from smart_classroom.thumbnail import save_thumbnail
from smart_classroom.storage import Storage
from datetime import datetime


def test_save_thumbnail_and_db_link(tmp_path):
    # create a fake BGR image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (50, 100, 150)

    thumb_dir = tmp_path / 'thumbs'
    thumb_dir.mkdir()
    ts = datetime.utcnow()
    path = save_thumbnail(img, out_dir=thumb_dir, timestamp=ts)
    assert path
    p = Path(path)
    assert p.exists()

    # use a temporary DB
    dbfile = tmp_path / 'db.sqlite'
    s = Storage(dbfile)
    eid = s.add_event('medium', 'thumb test', {'low':0.1,'medium':0.8,'high':0.1}, timestamp=ts, thumbnail_path=str(p))
    events = s.get_events()
    assert any(e.get('thumbnail_path') == str(p) for e in events)

    aid = s.add_attendance('testperson', timestamp=ts, level='medium', confidence=0.8, thumbnail_path=str(p))
    atts = s.get_attendance()
    assert any(a.get('thumbnail_path') == str(p) for a in atts)

    s.close()
