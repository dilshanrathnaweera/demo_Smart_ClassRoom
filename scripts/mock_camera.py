"""Mock camera generator for testing and CI.

Generates synthetic frames (random noise or colored rectangles) to simulate a webcam.
"""
import numpy as np


def generate_frame(width=640, height=480, color=(128, 128, 128)):
    # simple solid-color image with a small moving rectangle for variety
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    t = np.random.randint(0, min(width, height) // 2)
    x = t
    y = t
    w = 50
    h = 30
    frame[y:y+h, x:x+w, :] = (255, 0, 0)
    return frame


if __name__ == '__main__':
    # quick demo print
    f = generate_frame()
    print('Generated frame shape:', f.shape)
