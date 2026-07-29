from pathlib import Path
from datetime import datetime
import uuid
import cv2
import numpy as np


def save_thumbnail(frame: np.ndarray, out_dir: Path = None, timestamp: datetime = None, max_size=(320,240)) -> str:
    """Save a small JPEG thumbnail for a frame and return the path string.

    - frame: BGR numpy array from OpenCV
    - out_dir: destination folder (defaults to data/thumbnails)
    - timestamp: used in filename; defaults to utcnow
    Returns the absolute path as string.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[1] / 'data' / 'thumbnails'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = (timestamp or datetime.utcnow()).strftime('%Y%m%dT%H%M%S')
    fname = f"thumb_{ts}_{uuid.uuid4().hex[:8]}.jpg"
    path = out_dir / fname

    try:
        # resize keeping aspect
        h, w = frame.shape[:2]
        max_w, max_h = max_size
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        # write JPEG
        cv2.imwrite(str(path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return str(path)
    except Exception:
        return ''


if __name__ == '__main__':
    print('thumbnail helper')
