"""
export_lstm_onnx.py — Export trained LSTM PyTorch model to ONNX.

Run from dl/:
    python prediction/export_lstm_onnx.py

Output: prediction/lstm.onnx
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from lstm_train import TrafficLSTM

PT_PATH   = Path(__file__).parent / "lstm_best.pt"
ONNX_PATH = Path(__file__).parent / "lstm.onnx"
WINDOW    = 30

model = TrafficLSTM()
model.load_state_dict(torch.load(PT_PATH, map_location="cpu"))
model.eval()

dummy = torch.zeros(1, WINDOW, 1)   # [batch, seq_len, features]

torch.onnx.export(
    model,
    dummy,
    str(ONNX_PATH),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=12,
)

print(f"Exported → {ONNX_PATH}")
