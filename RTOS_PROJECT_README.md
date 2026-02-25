# ⚙️ RTOS Project

## 1. Title
**Adaptive Traffic Control System (ATCS) with Real-Time Priority Scheduling and Emergency Response**

---

## 2. Overview

This project implements an Adaptive Traffic Control System (ATCS) using FreeRTOS on STM32H7. Unlike fixed-timer systems, ATCS adjusts signal timing based on real-time traffic density. The system acts as the "brain and hands" - it receives traffic information from Raspberry Pi, makes control decisions, and physically controls the LED traffic signals. It guarantees fast emergency response (<50ms) through interrupt handling and priority-based task scheduling.

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
- WeAct STM32H743VI - Main microcontroller
- RGB LEDs (6-8 units) - Traffic signals
- Push Button - Emergency trigger
- Breadboard & wires

**Software:**
- FreeRTOS - Real-time operating system
- STM32CubeIDE - Development environment
- ST-Link - Programming/debugging

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

**Connection: Raspberry Pi → STM32H7**

**Method:** UART (Serial)
- Baud rate: 115200 bps
- Wiring: RPi GPIO14 → STM32 PA10 (RX), RPi GPIO15 ← STM32 PA9 (TX), Ground connected
- Updates every 2-3 seconds
- RPi sends traffic data, STM32 receives and controls signals

---

## 7. Input/Output Data Format

**Input to STM32 (JSON from Raspberry Pi):**
```json
{
  "vehicle_counts": {
    "north": 15,
    "south": 8,
    "east": 22,
    "west": 12
  },
  "density_level": "HIGH"
}
```

**Output from STM32:**
- Physical: Controls 6 RGB LEDs (Red/Yellow/Green for North-South and East-West)
- States: LED ON (signal active) or OFF
- Emergency button input triggers hardware interrupt

---

## 8. Concepts Learned

**RTOS Fundamentals:**
- Task scheduling and priority management
- Preemptive vs cooperative scheduling
- Context switching

**Interrupt Handling:**
- Hardware interrupts and ISR design
- Fast response times (<50ms)
- ISR-to-task communication

**Inter-Process Communication:**
- Semaphores for signaling
- Mutexes for data protection
- Message queues for data passing

**Real-Time Constraints:**
- Deterministic timing
- Worst-case execution time (WCET)
- Meeting deadlines

**Hardware Integration:**
- GPIO control for LEDs
- UART serial communication
- Button input handling

**System Reliability:**
- Watchdog timers
- Fail-safe mechanisms
- Error handling

---

## 🔗 Useful Resources

**RTOS Frameworks:**
- FreeRTOS: https://www.freertos.org/
- Zephyr RTOS: https://zephyrproject.org/
- CMSIS-RTOS: https://arm-software.github.io/CMSIS_5/

**Development Boards:**
- STM32 Nucleo: https://www.st.com/en/evaluation-tools/stm32-nucleo-boards.html
- WeAct Studio STM32: https://github.com/WeActStudio

**Development Tools:**
- STM32CubeIDE: https://www.st.com/en/development-tools/stm32cubeide.html
- SEGGER SystemView: https://www.segger.com/products/development-tools/systemview/

**Learning Resources:**
- FreeRTOS Tutorials: https://www.freertos.org/tutorial/
- Embedded Systems Course: https://www.edx.org/learn/embedded-systems
- Real-Time Systems Book: "Real-Time Systems" by Jane W. S. Liu

---

## 📚 References

1. Richard Barry, "Mastering the FreeRTOS Real Time Kernel", 2016
2. Giorgio Buttazzo, "Hard Real-Time Computing Systems", Springer, 2011
3. Liu & Layland, "Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment", JACM 1973
4. STM32H7 Reference Manual, STMicroelectronics
5. ARM Cortex-M7 Technical Reference Manual, ARM Limited
