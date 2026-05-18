# AI CONTEXT

**READ THIS FIRST IF YOU ARE AN AI BOT!**

## LOGGING RULE (CRITICAL)
- IF YOU MAKE CODE CHANGES, YOU MUST LOG IT HERE AT BOTTOM OF FILE.
- KEEP LOG VERY SHORT. Caveman english ok. Date + what changed + files touched. NO ESSAYS. Keep file small.

## WHAT IS THIS PROJECT
Adaptive Traffic Control System (ATCS). 
Goal: Count cars using AI, predict traffic, change traffic lights dynamically.

## HOW IT WORKS (2 PARTS)

**1. Deep Learning (`dl/`) - Runs on Raspberry Pi (Python)**
- YOLOv8 detects cars in video stream.
- LSTM predicts traffic density (LOW, MED, HIGH, CONGESTED).
- Sends data as JSON over UART to ESP32.

**2. Hardware (`rtos/`) - Runs on ESP32 (FreeRTOS, C++)**
- Changes LED traffic lights based on traffic density.
- **Multi-Junction Serial Setup (Master/Slave):**
  - `master` (City Centre): Gets UART data from deep learning python script. Controls city lights. Forwards density info over Serial wire to the outskirt board.
  - `slave` (Outskirt): Listens to Serial wire. If city is CONGESTED/HIGH, it changes its own timing (City Priority Mode) to stop feeding cars to city and let city clear out.
- **FreeRTOS Tasks:**
  - Prio 5: Emergency Handler (button interrupt)
  - Prio 4: Signal Controller (LEDs)
  - Prio 3: Density Reader (UART from Mac/RPi)
  - Prio 2: Junction Sync (Serial bridge)
  - Prio 1: Display Monitor (Serial log)

**3. Windows Automated Training Pipeline (`dl/fine_tuning/`)**
- Users can drop raw Roboflow ZIP files into `dataset_downloads/`.
- `auto_train_windows.py` will automatically extract, remap random classes to the Universal ATCS standard, merge datasets, train a YOLOv8 model on a Windows GPU, and export to `.onnx`.

## CHANGELOG
- [2026-05-06] Initial AI Context created. Planned NRF24L01 Master/Slave logic for c_esp32 and o_esp32.
- [2026-05-08] Wrote NRF Master/Slave code in rtos/atcs/atcs.ino. c_esp32 (Master) broadcasts density. o_esp32 (Slave) enters City Priority Mode and denies local emergency if city congested.
- [2026-05-18] Removed NRF hardware logic in favor of a direct Serial JSON bridge handled by `demo_runner.py`. Wrote `auto_train_windows.py` for automated local dataset merging and GPU training on Windows. Cleaned codebase of 150MB of unused `.pt` weights and deprecated scripts.
