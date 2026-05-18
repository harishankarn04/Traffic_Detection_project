#!/bin/bash
.venv/bin/pip install scikit-learn seaborn matplotlib
.venv/bin/python dl/prediction/extract_video_data.py
.venv/bin/python dl/prediction/lstm_train.py
.venv/bin/python dl/prediction/export_lstm_onnx.py
