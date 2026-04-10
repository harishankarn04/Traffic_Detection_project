/*
 * atcs.ino — Adaptive Traffic Control System
 * Platform: ESP32 (Arduino framework)
 *
 * 5 FreeRTOS Tasks:
 *   Priority 5 — Emergency Handler  (ISR + task)
 *   Priority 4 — Signal Controller  (adaptive timing)
 *   Priority 3 — Density Reader     (UART from RPi)
 *   Priority 2 — Junction Sync      (inter-junction coordination)
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

// ─── Pin Definitions ─────────────────────────────────────────────────────────
#define NS_RED    25
#define NS_YELLOW 26
#define NS_GREEN  27
#define EW_RED    14
#define EW_YELLOW 12
#define EW_GREEN  13
#define EMERGENCY_BTN 0   // Boot button on most DevKit boards

// ─── Adaptive Timing (seconds) ───────────────────────────────────────────────
#define TIME_LOW       17
#define TIME_MEDIUM    30
#define TIME_HIGH      45
#define TIME_CONGESTED 60
#define TIME_YELLOW     3
#define TIME_EMERGENCY_HOLD 30

// ─── Shared Data (protected by mutex) ────────────────────────────────────────
typedef struct {
    int    vehicle_count;
    char   current_density[12];
    char   predicted_density[12];
    bool   emergency_active;
} TrafficData_t;

TrafficData_t trafficData = {0, "LOW", "LOW", false};

// ─── FreeRTOS Handles ────────────────────────────────────────────────────────
SemaphoreHandle_t xMutex;
SemaphoreHandle_t xEmergencySem;
QueueHandle_t     xDensityQueue;

// ─── Signal State ────────────────────────────────────────────────────────────
typedef enum { NS_GO, NS_YELLOW_PHASE, EW_GO, EW_YELLOW_PHASE } SignalState_t;
volatile SignalState_t signalState = NS_GO;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
void setSignal(bool ns_red, bool ns_yellow, bool ns_green,
               bool ew_red, bool ew_yellow, bool ew_green) {
    digitalWrite(NS_RED,    ns_red);
    digitalWrite(NS_YELLOW, ns_yellow);
    digitalWrite(NS_GREEN,  ns_green);
    digitalWrite(EW_RED,    ew_red);
    digitalWrite(EW_YELLOW, ew_yellow);
    digitalWrite(EW_GREEN,  ew_green);
}

void allRed() {
    setSignal(HIGH, LOW, LOW, HIGH, LOW, LOW);
}

int getGreenTime(const char* density) {
    if      (strcmp(density, "CONGESTED") == 0) return TIME_CONGESTED;
    else if (strcmp(density, "HIGH")      == 0) return TIME_HIGH;
    else if (strcmp(density, "MEDIUM")    == 0) return TIME_MEDIUM;
    else                                         return TIME_LOW;
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 1 — Emergency Handler (Priority 5)
// ─────────────────────────────────────────────────────────────────────────────
void IRAM_ATTR emergencyISR() {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xSemaphoreGiveFromISR(xEmergencySem, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

void taskEmergencyHandler(void* pvParams) {
    while (true) {
        if (xSemaphoreTake(xEmergencySem, portMAX_DELAY) == pdTRUE) {
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
void taskSignalController(void* pvParams) {
    char density[12];
    int greenTime;

    while (true) {
        // Skip if emergency active
        if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            bool emergency = trafficData.emergency_active;
            strncpy(density, trafficData.current_density, sizeof(density));
            xSemaphoreGive(xMutex);

            if (emergency) {
                vTaskDelay(pdMS_TO_TICKS(500));
                continue;
            }
        }

        greenTime = getGreenTime(density);

        // NS green phase
        signalState = NS_GO;
        setSignal(LOW, LOW, HIGH, HIGH, LOW, LOW);
        Serial.printf("[SIGNAL] NS GREEN for %ds (density: %s)\n", greenTime, density);
        vTaskDelay(pdMS_TO_TICKS(greenTime * 1000));

        // NS yellow phase
        signalState = NS_YELLOW_PHASE;
        setSignal(LOW, HIGH, LOW, HIGH, LOW, LOW);
        vTaskDelay(pdMS_TO_TICKS(TIME_YELLOW * 1000));

        // Re-read density for EW phase
        if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            strncpy(density, trafficData.current_density, sizeof(density));
            xSemaphoreGive(xMutex);
        }
        greenTime = getGreenTime(density);

        // EW green phase
        signalState = EW_GO;
        setSignal(HIGH, LOW, LOW, LOW, LOW, HIGH);
        Serial.printf("[SIGNAL] EW GREEN for %ds (density: %s)\n", greenTime, density);
        vTaskDelay(pdMS_TO_TICKS(greenTime * 1000));

        // EW yellow phase
        signalState = EW_YELLOW_PHASE;
        setSignal(HIGH, LOW, LOW, LOW, HIGH, LOW);
        vTaskDelay(pdMS_TO_TICKS(TIME_YELLOW * 1000));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 3 — Density Reader (Priority 3)
// Reads JSON from RPi via UART2 every 2s
// ─────────────────────────────────────────────────────────────────────────────
void taskDensityReader(void* pvParams) {
    String line;

    // Flush any garbage on the line before reading valid JSON
    vTaskDelay(pdMS_TO_TICKS(2000));
    while (Serial2.available()) Serial2.read();

    while (true) {
        if (Serial2.available()) {
            line = Serial2.readStringUntil('\n');
            line.trim();

            // Skip lines that don't look like JSON
            if (line.length() == 0 || line[0] != '{') continue;

            StaticJsonDocument<256> doc;
            DeserializationError err = deserializeJson(doc, line);

            if (!err) {
                bool emergency_from_cam = doc["emergency"] | false;

                // Trigger emergency semaphore if camera detected emergency vehicle
                if (emergency_from_cam) {
                    xSemaphoreGive(xEmergencySem);
                }

                if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                    trafficData.vehicle_count = doc["vehicle_count"] | 0;

                    const char* cd = doc["current_density"] | "LOW";
                    strncpy(trafficData.current_density, cd, sizeof(trafficData.current_density));

                    const char* pd = doc["predicted_density"] | "LOW";
                    strncpy(trafficData.predicted_density, pd, sizeof(trafficData.predicted_density));

                    xSemaphoreGive(xMutex);
                    Serial.printf("[UART] count=%d current=%s predicted=%s emergency=%s\n",
                        trafficData.vehicle_count,
                        trafficData.current_density,
                        trafficData.predicted_density,
                        emergency_from_cam ? "YES" : "NO");
                }
            } else {
                Serial.printf("[UART] Parse error: %s\n", line.c_str());
            }
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 4 — Junction Sync (Priority 2)
// Placeholder for multi-junction coordination via queue
// ─────────────────────────────────────────────────────────────────────────────
void taskJunctionSync(void* pvParams) {
    while (true) {
        // In a multi-junction setup: receive state from neighbouring junctions,
        // coordinate green wave timing via xQueueSend/Receive
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 5 — Display Monitor (Priority 1)
// ─────────────────────────────────────────────────────────────────────────────
void taskDisplayMonitor(void* pvParams) {
    const char* stateNames[] = {"NS_GO", "NS_YELLOW", "EW_GO", "EW_YELLOW"};

    while (true) {
        if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            Serial.printf("[MONITOR] count=%d | current=%s | predicted=%s | signal=%s | emergency=%s\n",
                trafficData.vehicle_count,
                trafficData.current_density,
                trafficData.predicted_density,
                stateNames[signalState],
                trafficData.emergency_active ? "YES" : "NO");
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
    Serial2.begin(115200, SERIAL_8N1, 16, 17);  // RX=16, TX=17

    // LED pins
    pinMode(NS_RED,    OUTPUT);
    pinMode(NS_YELLOW, OUTPUT);
    pinMode(NS_GREEN,  OUTPUT);
    pinMode(EW_RED,    OUTPUT);
    pinMode(EW_YELLOW, OUTPUT);
    pinMode(EW_GREEN,  OUTPUT);
    allRed();

    // Emergency button
    pinMode(EMERGENCY_BTN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(EMERGENCY_BTN), emergencyISR, FALLING);

    // FreeRTOS primitives
    xMutex        = xSemaphoreCreateMutex();
    xEmergencySem = xSemaphoreCreateBinary();
    xDensityQueue = xQueueCreate(5, sizeof(TrafficData_t));

    // Create tasks
    xTaskCreatePinnedToCore(taskEmergencyHandler,  "Emergency",  2048, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(taskSignalController,  "Signal",     4096, NULL, 4, NULL, 1);
    xTaskCreatePinnedToCore(taskDensityReader,     "Density",    4096, NULL, 3, NULL, 1);
    xTaskCreatePinnedToCore(taskJunctionSync,      "Sync",       2048, NULL, 2, NULL, 0);
    xTaskCreatePinnedToCore(taskDisplayMonitor,    "Monitor",    2048, NULL, 1, NULL, 0);

    Serial.println("[ATCS] System started.");
}

void loop() {
    // Empty — FreeRTOS scheduler handles everything
    vTaskDelay(portMAX_DELAY);
}
