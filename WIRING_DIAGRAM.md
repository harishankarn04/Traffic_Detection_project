# Wiring Diagram — ATCS Hardware Setup

## Components Needed

| Component | Quantity |
|-----------|----------|
| ESP32 DevKit V1 | 1 |
| Raspberry Pi 4 | 1 |
| Red LED | 2 |
| Yellow LED | 2 |
| Green LED | 2 |
| 330Ω resistor | 6 |
| Push button | 1 |
| Breadboard | 1 |
| Jumper wires | ~20 |

---

## LED Colours

| Signal | GPIO | LED Colour |
|--------|------|------------|
| NS_RED | 25 | Red |
| NS_YELLOW | 26 | Yellow |
| NS_GREEN | 27 | Green |
| EW_RED | 14 | Red |
| EW_YELLOW | 12 | Yellow |
| EW_GREEN | 13 | Green |

NS = North-South direction, EW = East-West direction.

---

## LED Wiring (per LED)

```
ESP32 GPIO ──[330Ω resistor]── LED anode (+, longer leg) ── LED cathode (−, shorter leg) ── GND rail
```

All 6 LED cathodes share the same GND rail on the breadboard.

---

## Emergency Button Wiring

```
ESP32 GPIO0 ──[push button]── GND rail
```

GPIO0 is the BOOT button already on the DevKit board — no external button needed unless you want one.

---

## RPi ↔ ESP32 UART Wiring

Both boards are 3.3V — **no level shifter needed**.

| RPi Pin | RPi GPIO | Direction | ESP32 Pin |
|---------|----------|-----------|-----------|
| Pin 8 | GPIO14 (TX) | → | GPIO16 (RX) |
| Pin 10 | GPIO15 (RX) | ← | GPIO17 (TX) |
| Pin 6 | GND | ── | GND |

> Cross the data lines: RPi TX → ESP32 RX, RPi RX ← ESP32 TX. Always share GND.

---

## Power

Power ESP32 from RPi USB port:
```
RPi USB-A port ──[USB cable]── ESP32 micro-USB
```
This shares GND automatically — no separate GND wire needed between boards.

---

## Full Breadboard Layout

```
RPi                          ESP32 DevKit V1
────                         ───────────────
[Pin 8  TX] ─────────────►  [GPIO16 RX]
[Pin 10 RX] ◄─────────────  [GPIO17 TX]
[Pin 6 GND] ──────────────  [GND] ──── breadboard GND rail
[USB port]  ──────────────  [micro-USB 5V]

GPIO25 ──[330Ω]── RED LED   (NS) ── GND rail
GPIO26 ──[330Ω]── YELLOW LED (NS) ── GND rail
GPIO27 ──[330Ω]── GREEN LED (NS) ── GND rail
GPIO14 ──[330Ω]── RED LED   (EW) ── GND rail
GPIO12 ──[330Ω]── YELLOW LED (EW) ── GND rail
GPIO13 ──[330Ω]── GREEN LED (EW) ── GND rail

GPIO0  ──[push button]── GND rail   (optional, BOOT button on board works too)
```

---

## Serial Monitor Check (Arduino IDE)

After flashing, open Serial Monitor at **115200 baud**. Expected output:
```
[ATCS] System started.
[SIGNAL] NS GREEN for 17s (density: LOW)
[MONITOR] count=0 | current=LOW | predicted=LOW | signal=NS_GO | emergency=NO
```
