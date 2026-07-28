import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Dict


class OnnxClassifier:
    """Loads an ONNX model (Azure Custom Vision) and runs inference.

    Expects:
    - input tensor name: "data"
    - input: 224x224 RGB
    - output: raw class scores in model's first output
    """

    def __init__(self, model_path: str, class_names: List[str] = None, input_name: str = "data"):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        # force input name to 'data' per requirements
        self.input_name = input_name
        self.input_shape = (1, 3, 224, 224)
        self.class_names = class_names or ["low", "medium", "high"]

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        # frame is BGR from OpenCV; convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        # transpose to NCHW
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        return arr

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def predict(self, frame: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Run inference on a single OpenCV BGR frame.

        Returns:
            label: predicted class label
            confidences: dict[class_name -> confidence]
        """
        inp = self._preprocess(frame)
        # ONNX runtime expects inputs as {input_name: array}
        outputs = self.session.run(None, {self.input_name: inp})
        scores = outputs[0]
        probs = self._softmax(scores).ravel()
        confidences = {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        best_idx = int(np.argmax(probs))
        return self.class_names[best_idx], confidences


if __name__ == "__main__":
    # simple CLI quick test (requires webcam and model path)
    import time
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <model.onnx>")
        sys.exit(1)

    model = sys.argv[1]
    clf = OnnxClassifier(model)
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            label, conf = clf.predict(frame)
            print(label, conf)
            time.sleep(0.2)
    finally:
        cap.release()
