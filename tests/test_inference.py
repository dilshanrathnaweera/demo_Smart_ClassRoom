import numpy as np
import cv2
import onnxruntime as ort

from smart_classroom.inference import OnnxClassifier


class DummySession:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        # return a dummy score array for 3 classes
        return [np.array([[0.1, 0.7, 0.2]], dtype=np.float32)]


def test_preprocess_shape_and_range(monkeypatch):
    # prevent actual ONNX session creation
    monkeypatch.setattr(ort, "InferenceSession", DummySession)
    clf = OnnxClassifier(model_path="dummy.onnx")

    # generate a synthetic BGR frame (480x640)
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    arr = clf._preprocess(frame)

    assert arr.shape == (1, 3, 224, 224)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_predict_returns_label_and_confidences(monkeypatch):
    monkeypatch.setattr(ort, "InferenceSession", DummySession)
    clf = OnnxClassifier(model_path="dummy.onnx")
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    label, conf = clf.predict(frame)

    assert label in clf.class_names
    assert isinstance(conf, dict)
    assert all(k in conf for k in clf.class_names)
    # probabilities should sum to ~1
    s = sum(conf.values())
    assert abs(s - 1.0) < 1e-3
