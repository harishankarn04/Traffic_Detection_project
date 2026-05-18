# Full System Wiring Diagram — ATCS 

This document details the complete wiring for **both** ESP32 units in the Master/Slave architecture.

---

## 1. `c_esp32` (City Centre / Master) Wiring
*This ESP32 is the brain. It connects to the Raspberry Pi to get AI data, controls the city traffic lights, and broadcasts to the outskirt.*

### A. Power & Ground
* Power the ESP32 by connecting it to the Raspberry Pi via a Micro-USB cable.
* Connect a GND pin from the ESP32 to the breadboard's negative (`-`) rail. **All components will share this GND rail.**

### B. Raspberry Pi 4 (UART Communication)
*No logic level shifter is needed; both are 3.3V logic.*
* **RPi Pin 8 (GPIO 14 - TX)** ──► **ESP32 GPIO 16 (RX)**
* **RPi Pin 10 (GPIO 15 - RX)** ◄── **ESP32 GPIO 17 (TX)**
* **RPi Pin 6 (GND)** ────────── **Breadboard GND rail**

### C. NRF24L01+PA+LNA (Wireless Radio)
* **VCC** ──► **ESP32 3.3V Pin** *(CRITICAL: Put a 10µF - 100µF capacitor across VCC and GND on the NRF module)*
* **GND** ──► **Breadboard GND rail**
* **CE**  ──► **ESP32 GPIO 4**
* **CSN** ──► **ESP32 GPIO 22**
* **SCK** ──► **ESP32 GPIO 18**
* **MOSI** ──► **ESP32 GPIO 23**
* **MISO** ──► **ESP32 GPIO 19**

### D. Traffic Light LEDs
*All LEDs require a 330Ω resistor between the ESP32 pin and the LED's longer leg (Anode). The shorter leg (Cathode) goes to the GND rail.*
* **RED** ──[330Ω]──► **ESP32 GPIO 25**
* **YELLOW** ──[330Ω]──► **ESP32 GPIO 26**
* **GREEN** ──[330Ω]──► **ESP32 GPIO 27**

### E. Emergency Button
*You can use the built-in "BOOT" button on the ESP32 DevKit, or wire an external one.*
* **Push Button Leg 1** ──► **ESP32 GPIO 0**
* **Push Button Leg 2** ──► **Breadboard GND rail**

---

## 2. `o_esp32` (Outskirt / Slave) Wiring
*This ESP32 acts independently but listens to the city's broadcast to enter "City Priority Mode". It does NOT connect to the Raspberry Pi.*

### A. Power & Ground
* Power the ESP32 via a standard USB wall adapter or power bank.
* Connect a GND pin from the ESP32 to the breadboard's negative (`-`) rail. 

### B. NRF24L01+PA+LNA (Wireless Radio)
*(Wired exactly the same as the Master)*
* **VCC** ──► **ESP32 3.3V Pin** *(CRITICAL: Put a 10µF - 100µF capacitor across VCC and GND on the NRF module)*
* **GND** ──► **Breadboard GND rail**
* **CE**  ──► **ESP32 GPIO 4**
* **CSN** ──► **ESP32 GPIO 22**
* **SCK** ──► **ESP32 GPIO 18**
* **MOSI** ──► **ESP32 GPIO 23**
* **MISO** ──► **ESP32 GPIO 19**

### C. Traffic Light LEDs
*(Wired exactly the same as the Master)*
* **RED** ──[330Ω]──► **ESP32 GPIO 25**
* **YELLOW** ──[330Ω]──► **ESP32 GPIO 26**
* **GREEN** ──[330Ω]──► **ESP32 GPIO 27**

### D. Emergency Button
*(Wired exactly the same as the Master)*
* **Push Button Leg 1** ──► **ESP32 GPIO 0**
* **Push Button Leg 2** ──► **Breadboard GND rail**
