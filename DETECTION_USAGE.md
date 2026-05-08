# Detection Usage Guide

All commands run from inside `dl/`.

```bash
cd dl
```

---

## Basic Usage

### Local video file
```bash
python detection/detect_video_onnx.py \
  --video data/raw/your_video.mp4 \
  --model yolov8n_traffic.onnx
```

### YouTube live stream
```bash
python detection/detect_video_onnx.py \
  --video 'https://www.youtube.com/live/6dp-bvQ7RWo?si=bZeWGx-gAPDMEPIm' \
  --model yolov8n_traffic.onnx
```

---

## Flags

| Flag | Description |
|------|-------------|
| `--video` | Path to local video file or YouTube URL (required) |
| `--model` | Path to ONNX model file (required) |
| `--save` | Save annotated output video to same folder as input |
| `--lstm` | Path to LSTM ONNX model for density prediction |
| `--serial` | Serial port to send JSON to ESP32 (e.g. `/dev/serial0`) |
| `--headless` | Run without display window (use on RPi with VNC) |

---

## Common Combinations

### Test on laptop (no ESP32)
```bash
python detection/detect_video_onnx.py \
  --video 'https://www.youtube.com/live/6dp-bvQ7RWo?si=bZeWGx-gAPDMEPIm' \
  --model yolov8n_traffic.onnx
```

### Test on laptop and save output
```bash
python detection/detect_video_onnx.py \
  --video data/raw/your_video.mp4 \
  --model yolov8n_traffic.onnx \
  --save
```

### RPi with ESP32 (headless)
```bash
python detection/detect_video_onnx.py \
  --video 'your_stream_url' \
  --model yolov8n_traffic.onnx \
  --serial /dev/serial0 \
  --headless
```

### RPi with ESP32 + LSTM prediction
```bash
python detection/detect_video_onnx.py \
  --video 'your_stream_url' \
  --model yolov8n_traffic.onnx \
  --lstm lstm.onnx \
  --serial /dev/serial0 \
  --headless
```

---

## Exporting ONNX from best.pt

Run from `dl/` after training completes.

### 640px — full accuracy (laptop/Mac)
```bash
python -c "from ultralytics import YOLO; YOLO('fine_tuning/runs/yolov8n_traffic/weights/best.pt').export(format='onnx', imgsz=640, simplify=True)"
```
Rename output to `yolov8n_traffic.onnx`.

### 320px — faster inference (RPi)
```bash
python -c "from ultralytics import YOLO; YOLO('fine_tuning/runs/yolov8n_traffic/weights/best.pt').export(format='onnx', imgsz=320, simplify=True)"
```
Rename output to `yolov8n_traffic_320.onnx`.

> **Important:** `INPUT_SIZE` in `detect_video_onnx.py` must match the model's export size.
> 640 model → `INPUT_SIZE = 640`, 320 model → `INPUT_SIZE = 320`

### INT8 Quantization — faster on RPi CPU (optional)

Converts float32 weights → INT8. ~2x faster, ~4x smaller, ~1-2% accuracy drop. No code changes needed.

Run on Mac/laptop after exporting ONNX:
```bash
python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('yolov8n_traffic.onnx', 'yolov8n_traffic_int8.onnx', weight_type=QuantType.QUInt8)
"
```

Copy `yolov8n_traffic_int8.onnx` to RPi and use with `--model yolov8n_traffic_int8.onnx`. `INPUT_SIZE` stays at 640.

---

## Setting Up on Raspberry Pi

Model files are gitignored (too large for git) — copy them manually via USB or SCP:

| File | Copy to |
|------|---------|
| `dl/yolov8n_traffic.onnx` | `dl/yolov8n_traffic.onnx` on RPi |
| `dl/lstm.onnx` | `dl/lstm.onnx` on RPi (optional, only when LSTM training done) |

Everything else comes via `git clone`:
```bash
git clone https://github.com/harishankarn04/Traffic_Detection_project.git
cd Traffic_Detection_project
pip install -r dl/requirements_rpi.txt
```

---

## Notes

- Press `q` to quit when display is shown
- Use `--headless` on RPi with VNC (no physical display)
- Videos can be placed in `dl/data/raw/` (gitignored, won't be committed)
- `yt-dlp` must be installed for YouTube streams: `pip install yt-dlp`
- ONNX provider is auto-selected: CoreML on Mac, CPU on RPi
