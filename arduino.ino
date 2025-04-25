/*
  * Combined Soil Moisture Sensor and DHT11 Temperature/Humidity Sensor
  * Using Adafruit DHT library for more reliable readings
  * With servo motor control based on Raspberry Pi predictions
  */

#include <DHT11.h>
#include <Servo.h>  // Add Servo library

// Pin definitions
const int analogSoilPin = A0;    // Analog output from soil sensor
const int digitalSoilPin = 2;    // Digital output from soil sensor
#define DHTPIN 8                 // DHT11 data pin
#define DHTTYPE DHT11            // DHT sensor type (DHT11)
#define SERVOPIN 9               // Servo motor control pin

// Objects
DHT11 dht(DHTPIN);
Servo irrigationServo;           // Create servo object

// Variables
int soilMoistureValue = 0;       // Variable to store analog soil moisture value
int currentPrediction = 0;       // Current irrigation prediction (0=no, 1=yes)
bool isIrrigating = false;       // Track if currently irrigating
String inputBuffer = "";         // Buffer for incoming serial data

void setup() {
  Serial.begin(9600);            // Initialize serial communication
  pinMode(digitalSoilPin, INPUT); // Set digital soil pin as input
  
  // Setup servo
  irrigationServo.attach(SERVOPIN);
  irrigationServo.write(90);     // Initialize servo to neutral position
  
  Serial.println("Combined Soil Moisture and DHT11 Sensor Test");
  Serial.println("-------------------------------------------");
  delay(2000);
}

void loop() {
  // Check for predictions from Raspberry Pi
  checkForPrediction();
  
  // ---- Soil Moisture Sensor Reading ----
  // Read the analog value from soil moisture sensor
  soilMoistureValue = analogRead(analogSoilPin);
  
  // Print the analog value to serial monitor
  Serial.print("Soil Moisture Value: ");
  Serial.print(soilMoistureValue);
  
  // Convert analog value to percentage (adjust these values based on your sensor's range)
  int moisturePercentage = map(soilMoistureValue, 1023, 0, 0, 100);
  
  // Keep percentage within 0-100 range
  moisturePercentage = constrain(moisturePercentage, 0, 100);
  
  Serial.print(" (");
  Serial.print(moisturePercentage);
  Serial.println("%)");
  
  // ---- DHT11 Temperature and Humidity Reading ----
  // Adding a small delay to ensure DHT sensor is ready
  delay(1000);
  
  // Reading temperature and humidity
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  // Check if reading was successful
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("Failed to read from DHT sensor!");
  } else {
    Serial.print("Humidity (%): ");
    Serial.println(humidity);
    Serial.print("Temperature (C): ");
    Serial.println(temperature);
  }
  
  Serial.println("-------------------------------------------");
  
  // Wait before next reading
  delay(2000);  // 2 second delay between readings
}

// Function to check for predictions from Raspberry Pi
void checkForPrediction() {
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    
    // Process complete message when newline received
    if (inChar == '\n') {
      // Check for prediction message format
      if (inputBuffer.startsWith("PREDICTION:")) {
        // Extract the prediction value
        int prediction = inputBuffer.substring(11).toInt();
        
        // Process the prediction
        if (prediction == 1 && currentPrediction == 0) {
          // Prediction changed from 0 to 1 - start irrigation
          currentPrediction = 1;
          irrigationServo.write(180);  // Turn servo to irrigation position
          isIrrigating = true;
          Serial.println("IRRIGATION: STARTED");
        }
        else if (prediction == 0 && currentPrediction == 1) {
          // Prediction changed from 1 to 0 - stop irrigation
          currentPrediction = 0;
          irrigationServo.write(90);   // Return servo to neutral position
          isIrrigating = false;
          Serial.println("IRRIGATION: STOPPED");
        }
      }
      inputBuffer = ""; // Clear buffer for next message
    } else {
      // Add character to buffer
      inputBuffer += inChar;
    }
  }
}
