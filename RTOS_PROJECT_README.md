# RTOS Project

## 1. Title
**Adaptive Traffic Control System (ATCS) with Real-Time Priority Scheduling and Emergency Response**

---

## 2. Overview

This project implements an Adaptive Traffic Control System (ATCS) using FreeRTOS on ESP32. Unlike fixed-timer systems, ATCS adjusts signal timing based on real-time traffic density. The system acts as the "brain and hands" - it receives traffic information from Raspberry Pi, makes control decisions, and physically controls the LED traffic signals. It guarantees fast emergency response (<50ms) through interrupt handling and priority-based task scheduling.

---

## 3. Objectives

- Implement ATCS with adaptive signal timing based on traffic density
- Build FreeRTOS system with 5 priority-based tasks
- Create emergency interrupt handler with <50ms response time
- Develop inter-task communication using semaphores, mutexes, and queues
- Control physical LED traffic signals
- Implement watchdog timer for system reliability

---

## 4. Tools Used

**Hardware:**
- ESP32 DevKit - Main microcontroller
- RGB LEDs (6-8 units) - Traffic signals
- Push Button - Emergency trigger
- Breadboard & wires

**Software:**
- FreeRTOS (via ESP-IDF) - Real-time operating system
- ESP-IDF - Development framework
- VS Code + ESP-IDF extension - Development environment

---

## 5. Technology Stack

**FreeRTOS with 5 Tasks (Priority High to Low):**

1. **Emergency Handler (Priority 5)** - Hardware interrupt, responds in <50ms
2. **ATCS Signal Controller (Priority 4)** - Manages signal states, calculates adaptive timing
3. **Density Sensor Reader (Priority 3)** - Reads data from RPi via UART every 2 seconds
4. **Multi-Junction Sync (Priority 2)** - Coordinates with other intersections
5. **Display & Monitor (Priority 1)** - Logs status via serial

**ATCS Adaptive Timing:**
- LOW density: 15-20s green
- MEDIUM: 25-35s green
- HIGH: 40-50s green
- CONGESTED: 55-70s green

**Synchronization:**
- Semaphore: Emergency signal from button to task
- Mutex: Protects shared traffic density data
- Message Queue: Inter-junction communication

**Safety Features:**
- Watchdog timer monitors system health
- Fail-safe mode: All RED if system fails

---

## 6. Communication Between Boards

**Connection: Raspberry Pi → ESP32**

**Method:** UART (Serial)
- Baud rate: 115200 bps
- Wiring: RPi GPIO14 (TX) → ESP32 GPIO16 (RX), RPi GPIO15 (RX) ← ESP32 GPIO17 (TX), Ground connected
- Updates every 2-3 seconds
- RPi sends traffic data, ESP32 receives and controls signals

---

## 7. Input/Output Data Format

**Input to ESP32 (JSON from Raspberry Pi):**
```json
{
  "vehicle_count": 15,
  "current_density": "HIGH",
  "predicted_density": "CONGESTED"
}
```

**Output from ESP32:**
- Physical: Controls 6 RGB LEDs (Red/Yellow/Green for North-South and East-West)
- States: LED ON (signal active) or OFF
- Emergency button input triggers hardware interrupt

---

## 8. Concepts Demonstrated

**RTOS Fundamentals:**
- Task scheduling and priority management
- Preemptive scheduling
- Context switching

**Interrupt Handling:**
- Hardware interrupts and ISR design
- Fast response times (<50ms)
- ISR-to-task communication via semaphore

**Inter-Process Communication:**
- Semaphores for signaling
- Mutexes for data protection
- Message queues for data passing

**Real-Time Constraints:**
- Deterministic timing
- Adaptive green time based on density

**Hardware Integration:**
- GPIO control for LEDs
- UART serial communication
- Button input handling

**System Reliability:**
- Watchdog timers
- Fail-safe mechanisms

---

## 9. Project Structure

```
rtos/
├── main/
│   ├── main.c                  # Entry point, task creation, FreeRTOS scheduler
│   ├── emergency_handler.c     # Priority 5 — ISR + emergency task
│   ├── signal_controller.c     # Priority 4 — adaptive signal timing
│   ├── density_reader.c        # Priority 3 — UART RX from RPi, JSON parse
│   ├── junction_sync.c         # Priority 2 — multi-junction coordination
│   └── display_monitor.c       # Priority 1 — serial logging
├── components/
│   └── uart_parser/            # Lightweight JSON parser for density data
└── CMakeLists.txt
```

---

## Useful Resources

- ESP-IDF FreeRTOS: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/system/freertos.html
- ESP32 GPIO: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/gpio.html
- ESP32 UART: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/uart.html
- FreeRTOS Official Docs: https://www.freertos.org/
