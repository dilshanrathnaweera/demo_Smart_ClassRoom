import cv2
import numpy as np
from pathlib import Path
import queue

from smart_classroom.dashboard import process_video_file
from smart_classroom.storage import Storage
from smart_classroom.attendance import AttendanceController


class FakeClassifier:
    def __init__(self):
        pass

    def predict(self, frame):
        # decide label based on mean green channel
        g = frame[:, :, 1].mean()
        if g < 50:
            return 'low', {'low': 0.9, 'medium': 0.05, 'high': 0.05}
        elif g < 150:
            return 'medium', {'low': 0.1, 'medium': 0.8, 'high': 0.1}
        else:
            return 'high', {'low': 0.05, 'medium': 0.05, 'high': 0.9}


def make_test_video(path: Path, frames=6, size=(320, 240)):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(path), fourcc, 5.0, (size[0], size[1]))
    for i in range(frames):
        if i < 2:
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8) + np.array([10, 10, 10], dtype=np.uint8)
        elif i < 4:
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8) + np.array([50, 120, 50], dtype=np.uint8)
        else:
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8) + np.array([200, 200, 200], dtype=np.uint8)
        out.write(img)
    out.release()


def test_process_uploaded_video(tmp_path):
    vid = tmp_path / 'test.avi'
    make_test_video(vid)

    db = tmp_path / 'db.sqlite'
    storage = Storage(db)
    attendance = AttendanceController(required_duration_seconds=1, confidence_threshold=0.5, absence_duration_seconds=1, person='uploader')
    thumbs = tmp_path / 'thumbs'
    thumbs.mkdir()

    clf = FakeClassifier()

    # process synchronously using storage
    process_video_file(str(vid), clf, storage=storage, attendance=attendance, thumbnail_dir=str(thumbs), frame_delay=0.01)

    events = storage.get_events()
    occ = storage.get_occupancy_history()
    # expect events and occupancy samples written
    assert len(events) >= 1
    assert len(occ) >= 1
    # thumbnails created
    thumbs_files = list(thumbs.glob('*'))
    assert len(thumbs_files) >= 1
