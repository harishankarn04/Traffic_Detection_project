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
- **Multi-Junction NRF24L01 Setup (Master/Slave):**
  - `c_esp32` (City Centre, Master): Gets UART data from RPi. Controls city lights. Broadcasts city traffic density over NRF radio.
  - `o_esp32` (Outskirt, Slave): Listens to NRF radio. If city is CONGESTED/HIGH, it changes its own timing (City Priority Mode) to stop feeding cars to city and let city clear out.
- **FreeRTOS Tasks:**
  - Prio 5: Emergency Handler (button interrupt)
  - Prio 4: Signal Controller (LEDs)
  - Prio 3: Density Reader (UART from RPi)
  - Prio 2: Junction Sync (NRF radio)
  - Prio 1: Display Monitor (Serial log)

## CHANGELOG
- [2026-05-06] Initial AI Context created. Planned NRF24L01 Master/Slave logic for c_esp32 and o_esp32.
- [2026-05-08] Wrote NRF Master/Slave code in rtos/atcs/atcs.ino. c_esp32 (Master) broadcasts density. o_esp32 (Slave) enters City Priority Mode and denies local emergency if city congested.
