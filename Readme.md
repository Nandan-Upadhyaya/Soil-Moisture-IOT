Here’s a simple Arduino sketch for this kind of project — it reads the soil moisture and if the soil is dry, the servo moves (e.g., simulating a watering action):#include <Servo.h>

const int soilMoisturePin = A0;  // Analog pin for soil sensor
const int threshold = 500;       // Moisture threshold
Servo myServo;

void setup() {
  Serial.begin(9600);
  myServo.attach(9);  // Attach servo to digital pin 9
  myServo.write(0);   // Initial position
}

void loop() {
  int moistureLevel = analogRead(soilMoisturePin);
  Serial.print("Soil Moisture Level: ");
  Serial.println(moistureLevel);

  if (moistureLevel < threshold) {
    Serial.println("Soil is dry! Moving servo to water.");
    myServo.write(90);  // Move servo to pour water
    delay(3000);        // Simulate watering time
    myServo.write(0);   // Return servo to initial position
  } else {
    Serial.println("Soil is wet. No need to water.");
    myServo.write(0);   // Ensure servo stays in initial position
  }

  delay(5000); // Delay before the next read
}



for raspberry :


import serial

import time



# Adjust this to match your port (often /dev/ttyUSB0 or /dev/ttyACM0 for Arduino)

arduino_port = '/dev/ttyACM0'  

baud_rate = 9600



try:

    arduino = serial.Serial(arduino_port, baud_rate, timeout=1)

    time.sleep(2)  # Give time for the connection to settle



    print("Connected to Arduino on", arduino_port)



    while True:

        if arduino.in_waiting > 0:

            data = arduino.readline().decode('utf-8').strip()

            print(f"Received from Arduino: {data}")

        time.sleep(1)



except serial.SerialException:

    print(f"Could not connect to Arduino on port {arduino_port}")

except KeyboardInterrupt:

    print("Program terminated by user.")

finally:

    if 'arduino' in locals() and arduino.is_open:

        arduino.close()





Explanation:
The Raspberry Pi opens a serial connection to /dev/ttyACM0 (adjust if yours is different, e.g., /dev/ttyUSB0).

It listens for messages from the Arduino — like the "Soil Moisture Level: XYZ" string from your earlier Arduino sketch.

It prints whatever the Arduino sends.