"""
lstm_predict.py — LSTM inference module with rolling buffer.

Used by detect_video_onnx.py to predict upcoming traffic density.
"""

import numpy as np
import onnxruntime as ort

WINDOW    = 30
MAX_COUNT = 50.0
LABELS    = ["LOW", "MEDIUM", "HIGH", "CONGESTED"]


class LSTMPredictor:
    def __init__(self, model_path: str):
        ort.set_default_logger_severity(3)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.buffer  = []   # rolling window of raw vehicle counts

    def update(self, count: int):
        self.buffer.append(count)
        if len(self.buffer) > WINDOW:
            self.buffer.pop(0)

    def predict(self) -> str | None:
        """Returns predicted density label, or None if buffer not full yet."""
        if len(self.buffer) < WINDOW:
            return None

        seq = np.array(self.buffer, dtype=np.float32) / MAX_COUNT
        seq = np.clip(seq, 0.0, 1.0).reshape(1, WINDOW, 1)

        output = self.session.run(None, {"input": seq})[0]  # [1, 4]
        class_id = int(np.argmax(output))
        return LABELS[class_id]
