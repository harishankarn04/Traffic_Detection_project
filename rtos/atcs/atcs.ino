/*
 * atcs.ino — Adaptive Traffic Control System
 * Platform: ESP32 (Arduino framework)
 *
 * 5 FreeRTOS Tasks:
 *   Priority 5 — Emergency Handler  (ISR + task)
 *   Priority 4 — Signal Controller  (adaptive timing)
 *   Priority 3 — Density Reader     (UART from RPi)
 *   Priority 2 — Junction Sync      (NRF24L01 Master/Slave)
 *   Priority 1 — Display Monitor    (serial logging)
 *
 * UART from RPi: GPIO16 (RX), GPIO17 (TX), 115200 baud
 * Emergency button: GPIO0 (with interrupt)
 *
 * LED pins:
 *   North-South: RED=25, YELLOW=26, GREEN=27
 *   East-West:   RED=14, YELLOW=12, GREEN=13
 */

#include <Arduino.h>
#include <ArduinoJson.h>
#include <RF24.h>
#include <SPI.h>
#include <nRF24L01.h>

// ─── Configuration Toggle ────────────────────────────────────────────────────
// Set to 1 for c_esp32 (City Centre - Master)
// Set to 0 for o_esp32 (Outskirt - Slave)
#define IS_C_ESP32 1

// ─── Pin Definitions ─────────────────────────────────────────────────────────
#define NS_RED 25
#define NS_YELLOW 26
#define NS_GREEN 27
#define EMERGENCY_BTN 0 // Boot button on most DevKit boards

// NRF24L01 SPI Pins (VSPI default: SCK=18, MISO=19, MOSI=23)
#define CE_PIN 4
#define CSN_PIN 5

// ─── NRF24 Setup ─────────────────────────────────────────────────────────────
RF24 radio(CE_PIN, CSN_PIN);
const byte address[6] = "ATCS1";

typedef struct {
  char city_density[12];
  bool city_emergency;
} SyncPacket_t;

// ─── Adaptive Timing (seconds) ───────────────────────────────────────────────
#define TIME_LOW 17
#define TIME_MEDIUM 30
#define TIME_HIGH 45
#define TIME_CONGESTED 60
#define TIME_YELLOW 3
#define TIME_EMERGENCY_HOLD 30

// ─── Shared Data (protected by mutex) ────────────────────────────────────────
typedef struct {
  int vehicle_count;
  char current_density[12];
  char predicted_density[12];
  bool emergency_active;

  // Remote data from Master (used by Slave)
  char city_density[12];
  bool city_emergency;
  uint32_t last_sync_time;
} TrafficData_t;

TrafficData_t trafficData = {0, "LOW", "LOW", false, "LOW", false, 0};

// ─── FreeRTOS Handles ────────────────────────────────────────────────────────
SemaphoreHandle_t xMutex;
SemaphoreHandle_t xEmergencySem;
QueueHandle_t xDensityQueue;

// ─── Signal State ────────────────────────────────────────────────────────────
typedef enum { NS_GO, NS_YELLOW_PHASE, EW_GO, EW_YELLOW_PHASE } SignalState_t;
volatile SignalState_t signalState = NS_GO;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
void setSignal(bool ns_red, bool ns_yellow, bool ns_green) {
  digitalWrite(NS_RED, ns_red);
  digitalWrite(NS_YELLOW, ns_yellow);
  digitalWrite(NS_GREEN, ns_green);
}

void allRed() { setSignal(HIGH, LOW, LOW); }

int getGreenTime(const char *density) {
  if (strcmp(density, "CONGESTED") == 0)
    return TIME_CONGESTED;
  else if (strcmp(density, "HIGH") == 0)
    return TIME_HIGH;
  else if (strcmp(density, "MEDIUM") == 0)
    return TIME_MEDIUM;
  else
    return TIME_LOW;
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 1 — Emergency Handler (Priority 5)
// ─────────────────────────────────────────────────────────────────────────────
void IRAM_ATTR emergencyISR() {
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  xSemaphoreGiveFromISR(xEmergencySem, &xHigherPriorityTaskWoken);
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

void taskEmergencyHandler(void *pvParams) {
  while (true) {
    if (xSemaphoreTake(xEmergencySem, portMAX_DELAY) == pdTRUE) {
      bool allowEmergency = true;

#if !IS_C_ESP32
      // For outskirt, if city is congested or in emergency, deny local
      // emergency priority
      if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        if (trafficData.city_emergency ||
            strcmp(trafficData.city_density, "CONGESTED") == 0 ||
            strcmp(trafficData.city_density, "HIGH") == 0) {
          allowEmergency = false;
        }
        xSemaphoreGive(xMutex);
      }
#endif

      if (!allowEmergency) {
        Serial.println(
            "[EMERGENCY] Local push denied: City priority overrides.");
        continue;
      }

      Serial.println("[EMERGENCY] Triggered! All RED for 30s");
      allRed();

      if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        trafficData.emergency_active = true;
        xSemaphoreGive(xMutex);
      }

      vTaskDelay(pdMS_TO_TICKS(TIME_EMERGENCY_HOLD * 1000));

      if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        trafficData.emergency_active = false;
        xSemaphoreGive(xMutex);
      }

      Serial.println("[EMERGENCY] Cleared. Resuming normal operation.");
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 2 — Signal Controller (Priority 4)
// ─────────────────────────────────────────────────────────────────────────────
void taskSignalController(void *pvParams) {
  char density[12];
  int greenTimeNS;
  int greenTimeEW;

  while (true) {
    bool localEmergency = false;
    bool cityEmergency = false;
    char cityDensity[12] = "LOW";

    if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
      localEmergency = trafficData.emergency_active;
      cityEmergency = trafficData.city_emergency;
      strncpy(density, trafficData.current_density, sizeof(density));
      strncpy(cityDensity, trafficData.city_density, sizeof(cityDensity));
      xSemaphoreGive(xMutex);
    }

    if (localEmergency) {
      vTaskDelay(pdMS_TO_TICKS(500));
      continue;
    }

#if !IS_C_ESP32
    // Slave specific behavior
    if (cityEmergency) {
      Serial.println(
          "[SIGNAL] City is in EMERGENCY! Freezing Outskirt to ALL RED.");
      allRed();
      vTaskDelay(pdMS_TO_TICKS(1000));
      continue;
    }
#endif

// Calculate green times
#if IS_C_ESP32
    // Master relies strictly on its own density
    greenTimeNS = getGreenTime(density);
    greenTimeEW = getGreenTime(density);
#else
    // Slave adjusts based on city. Assuming NS feeds the city.
    greenTimeNS = getGreenTime(density);
    greenTimeEW = getGreenTime(density);

    if (strcmp(cityDensity, "HIGH") == 0 ||
        strcmp(cityDensity, "CONGESTED") == 0) {
      Serial.println("[SIGNAL] City Priority Mode active.");
      greenTimeNS = TIME_LOW; // Restrict flow into the city
      greenTimeEW =
          TIME_CONGESTED; // Allow cross traffic or flow away from city to clear
    }
#endif

    // NS green phase
    signalState = NS_GO;
    setSignal(LOW, LOW, HIGH); // Green
    Serial.printf("[SIGNAL] GREEN for %ds (local: %s)\n", greenTimeNS, density);
    vTaskDelay(pdMS_TO_TICKS(greenTimeNS * 1000));

    // NS yellow phase
    signalState = NS_YELLOW_PHASE;
    setSignal(LOW, HIGH, LOW); // Yellow
    vTaskDelay(pdMS_TO_TICKS(TIME_YELLOW * 1000));

    // Re-read before EW phase
    if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
      localEmergency = trafficData.emergency_active;
      xSemaphoreGive(xMutex);
    }
    if (localEmergency)
      continue;

    // EW green phase (Simulated cross-traffic, our light is RED)
    signalState = EW_GO;
    setSignal(HIGH, LOW, LOW); // Red
    Serial.printf("[SIGNAL] RED for %ds (Cross-traffic is green)\n",
                  greenTimeEW);
    vTaskDelay(pdMS_TO_TICKS(greenTimeEW * 1000));

    // EW yellow phase (Simulated cross-traffic yellow)
    signalState = EW_YELLOW_PHASE;
    setSignal(HIGH, LOW, LOW); // Red
    vTaskDelay(pdMS_TO_TICKS(TIME_YELLOW * 1000));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 3 — Density Reader (Priority 3)
// ─────────────────────────────────────────────────────────────────────────────
void taskDensityReader(void *pvParams) {
  String line;
  vTaskDelay(pdMS_TO_TICKS(2000));
  while (Serial2.available())
    Serial2.read();

  while (true) {
    if (Serial2.available()) {
      line = Serial2.readStringUntil('\n');
      line.trim();

      if (line.length() == 0 || line[0] != '{')
        continue;

      StaticJsonDocument<256> doc;
      DeserializationError err = deserializeJson(doc, line);

      if (!err) {
        bool emergency_from_cam = doc["emergency"] | false;
        if (emergency_from_cam) {
          xSemaphoreGive(xEmergencySem);
        }

        if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
          trafficData.vehicle_count = doc["vehicle_count"] | 0;
          strncpy(trafficData.current_density, doc["current_density"] | "LOW",
                  sizeof(trafficData.current_density));
          strncpy(trafficData.predicted_density,
                  doc["predicted_density"] | "LOW",
                  sizeof(trafficData.predicted_density));
          xSemaphoreGive(xMutex);
        }
      }
    }
    vTaskDelay(pdMS_TO_TICKS(100));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 4 — Junction Sync (Priority 2)
// NRF24L01 Master/Slave communication
// ─────────────────────────────────────────────────────────────────────────────
void taskJunctionSync(void *pvParams) {
  if (!radio.begin()) {
    Serial.println("[NRF] Radio hardware is not responding!");
    vTaskSuspend(NULL); // Suspend task if hardware fails
  }

  radio.setPALevel(RF24_PA_MAX);
  radio.setDataRate(RF24_250KBPS); // Better range, cuts through interference

#if IS_C_ESP32
                                   // Master Mode
  radio.openWritingPipe(address);
  radio.stopListening();
  SyncPacket_t packet;

  while (true) {
    if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
      strncpy(packet.city_density, trafficData.current_density,
              sizeof(packet.city_density));
      packet.city_emergency = trafficData.emergency_active;
      xSemaphoreGive(xMutex);

      bool ok = radio.write(&packet, sizeof(packet));
      if (!ok) {
        // Packet failed to send (no ack), normal in noisy environments
      }
    }
    vTaskDelay(pdMS_TO_TICKS(2000)); // Broadcast every 2 seconds
  }
#else
                                   // Slave Mode
  radio.openReadingPipe(0, address);
  radio.startListening();
  SyncPacket_t packet;

  while (true) {
    if (radio.available()) {
      radio.read(&packet, sizeof(packet));

      if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        strncpy(trafficData.city_density, packet.city_density,
                sizeof(trafficData.city_density));
        trafficData.city_emergency = packet.city_emergency;
        trafficData.last_sync_time = millis();
        xSemaphoreGive(xMutex);
      }
    }

    // Timeout check: If no data received from City in 10s, revert to default
    if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
      if (millis() - trafficData.last_sync_time > 10000) {
        strncpy(trafficData.city_density, "LOW",
                sizeof(trafficData.city_density));
        trafficData.city_emergency = false;
      }
      xSemaphoreGive(xMutex);
    }

    vTaskDelay(pdMS_TO_TICKS(100));
  }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 5 — Display Monitor (Priority 1)
// ─────────────────────────────────────────────────────────────────────────────
void taskDisplayMonitor(void *pvParams) {
  const char *stateNames[] = {"NS_GO", "NS_YELLOW", "EW_GO", "EW_YELLOW"};

  while (true) {
    if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
#if IS_C_ESP32
      Serial.printf("[MASTER] count=%d | current=%s | signal=%s | emerg=%s\n",
                    trafficData.vehicle_count, trafficData.current_density,
                    stateNames[signalState],
                    trafficData.emergency_active ? "YES" : "NO");
#else
      Serial.printf(
          "[SLAVE] local=%s | city=%s | signal=%s | emerg=%s(city:%s)\n",
          trafficData.current_density, trafficData.city_density,
          stateNames[signalState], trafficData.emergency_active ? "YES" : "NO",
          trafficData.city_emergency ? "YES" : "NO");
#endif
      xSemaphoreGive(xMutex);
    }
    vTaskDelay(pdMS_TO_TICKS(2000));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Setup & Loop
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, 16, 17); // RX=16, TX=17

  // LED pins
  pinMode(NS_RED, OUTPUT);
  pinMode(NS_YELLOW, OUTPUT);
  pinMode(NS_GREEN, OUTPUT);
  allRed();

  // Emergency button
  pinMode(EMERGENCY_BTN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(EMERGENCY_BTN), emergencyISR, FALLING);

  // FreeRTOS primitives
  xMutex = xSemaphoreCreateMutex();
  xEmergencySem = xSemaphoreCreateBinary();
  xDensityQueue = xQueueCreate(5, sizeof(TrafficData_t));

  // Create tasks
  xTaskCreatePinnedToCore(taskEmergencyHandler, "Emergency", 2048, NULL, 5,
                          NULL, 1);
  xTaskCreatePinnedToCore(taskSignalController, "Signal", 4096, NULL, 4, NULL,
                          1);
  xTaskCreatePinnedToCore(taskDensityReader, "Density", 4096, NULL, 3, NULL, 1);
  xTaskCreatePinnedToCore(taskJunctionSync, "Sync", 3072, NULL, 2, NULL,
                          0); // Increased stack for NRF
  xTaskCreatePinnedToCore(taskDisplayMonitor, "Monitor", 2048, NULL, 1, NULL,
                          0);

#if IS_C_ESP32
  Serial.println("[ATCS] c_esp32 (Master) System started.");
#else
  Serial.println("[ATCS] o_esp32 (Slave) System started.");
#endif
}

void loop() {
  // Empty — FreeRTOS scheduler handles everything
  vTaskDelay(portMAX_DELAY);
}
