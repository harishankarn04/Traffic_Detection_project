# 🚦 Vehicle Density Estimation System

## Deep Learning Traffic Project

------------------------------------------------------------------------

## 📌 Project Goal

Build an AI-based traffic system that:

1.  Detects vehicles from traffic frames
2.  Counts vehicles per frame
3.  Converts count → Density Level
4.  Stores time-series vehicle data
5.  Predicts future traffic density (LSTM)
6.  Sends density data to controller (RPi → STM32)

------------------------------------------------------------------------

# 🧠 System Pipeline

Frames → Vehicle Detection → Vehicle Count → Density Level\
↓\
Store Time-Series Data\
↓\
LSTM Prediction

------------------------------------------------------------------------

# 📁 Project Structure

    traffic-density-system/
    │
    ├── dataset/
    │   ├── images/
    │   ├── labels/
    │
    ├── detection/
    │   ├── train.py
    │   ├── detect.py
    │   └── best.pt
    │
    ├── density/
    │   ├── count_vehicles.py
    │   ├── density_logic.py
    │   └── vehicle_counts.csv
    │
    ├── prediction/
    │   ├── lstm_train.py
    │   ├── lstm_model.h5
    │   └── preprocess.py
    │
    ├── deployment/
    │   └── send_to_controller.py
    │
    └── README.md

------------------------------------------------------------------------

# 🔥 EXECUTION PATH (Follow Step-by-Step)

------------------------------------------------------------------------

## ✅ PHASE 1 --- Vehicle Detection

### Step 1: Prepare Dataset

-   Organize into:
    -   images/train
    -   images/val
    -   labels/train
    -   labels/val

Classes: - car - bus - truck - motorcycle

### Step 2: Train Detection Model

-   Use pretrained YOLO weights
-   Epochs: 50
-   Image size: 640
-   Batch size: 8--16

Target: - mAP@0.5 \> 80%

✔ Detection Phase Complete

------------------------------------------------------------------------

## ✅ PHASE 2 --- Vehicle Counting

### Step 3: Count Vehicles Per Frame

Example output: Frame_001 → 18 vehicles\
Frame_002 → 22 vehicles

Store in CSV:

timestamp,total_count\
10:01,18\
10:02,22

File location: density/vehicle_counts.csv

✔ Counting Phase Complete

------------------------------------------------------------------------

## ✅ PHASE 3 --- Convert Count → Density

Example threshold mapping:

0--10 → LOW\
11--25 → MEDIUM\
26--40 → HIGH\
40+ → CONGESTED

Example Output:

{ "timestamp": "10:02", "vehicle_count": 22, "density_level": "MEDIUM" }

✔ Current Density System Complete

------------------------------------------------------------------------

## ✅ PHASE 4 --- Prepare Data for LSTM

Sequence length = 30 timesteps

Example input: \[18, 22, 25, 28, 30, 32, 29, 27, ...\]

Output: Next density class

✔ Time-Series Dataset Ready

------------------------------------------------------------------------

## ✅ PHASE 5 --- Train LSTM

Architecture:

Input → LSTM(128)\
→ Dropout(0.3)\
→ LSTM(64)\
→ Dense\
→ Softmax (4 classes)

Train: - 50 epochs - Early stopping - Categorical crossentropy

Target: - Accuracy \> 85%

✔ Prediction Model Complete

------------------------------------------------------------------------

## ✅ PHASE 6 --- Full Integration

1.  Detect vehicles
2.  Count vehicles
3.  Calculate current density
4.  Update time-series buffer
5.  Predict next density
6.  Send JSON output

------------------------------------------------------------------------

# 📤 Final Output Format

{ "timestamp": "2026-02-13 14:30:45", "vehicle_count": 32,
"current_density": "HIGH", "predicted_density": "MEDIUM" }

------------------------------------------------------------------------

# 🎯 Milestone Checklist

Detection: - \[ \] Dataset prepared - \[ \] Model trained - \[ \] mAP
evaluated

Density: - \[ \] Counting script working - \[ \] Density classification
working - \[ \] CSV generation working

Prediction: - \[ \] LSTM dataset prepared - \[ \] LSTM trained - \[ \]
Accuracy evaluated

Deployment: - \[ \] JSON output structured - \[ \] Communication tested

------------------------------------------------------------------------

# 🏁 Suggested Timeline

Week 1 → Detection\
Week 2 → Counting + Density Logic\
Week 3 → LSTM Prediction\
Week 4 → Integration + Deployment

------------------------------------------------------------------------

## 📘 Notes

-   Detection accuracy impacts density accuracy.
-   Good annotations improve final performance.
-   Complete each phase before moving forward.
-   Keep system modular and structured.

------------------------------------------------------------------------

End of README
