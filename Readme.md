SmartSoil, a comprehensive IoT-based smart irrigation management system that leverages machine learning to optimize water usage in agriculture. This project seamlessly integrates hardware components, cloud services, and AI to create an end-to-end smart farming solution.

🌱 Project Overview
Designed and implemented an automated irrigation system that analyzes real-time soil moisture, temperature, and humidity data to make intelligent irrigation decisions tailored to specific crop requirements (supporting Coffee, Garden Flowers, Groundnuts, Maize, Paddy, Potato, Pulse, Sugarcane, and Wheat). Our system also allows the user to input the crop name and then based on the crop and the sensor data associated with that crop, predictions are made and processed accordingly.

Technical Architecture
Hardware Layer: Arduino microcontroller interfaced with soil moisture sensors, temperature/humidity sensors (DHT11), and servo motors for physical irrigation control
Processing Layer: Raspberry Pi running Python for data processing, ML inference, and system orchestration
Cloud Integration: Real-time and historical data storage using Firebase Realtime Database
ML Pipeline: K-Nearest Neighbors (KNN) classifier deployed as PMML model, stored on Google Drive and dynamically loaded to the edge device (Google Drive authentication using Pydrive with OAuth Credentials)

Additional Libraries used:
firebase-admin: Firebase data storage
pydrive: Google Drive API integration
pypmml: Processing PMML model files
pyserial: Serial communication with Arduino
pickle: Object serialization
DHT11 sensor library
Servo library

Cloud Services:
Firebase Realtime Database: Data storage & retrieval
Google Drive API: Remote model storage
Google Cloud OAuth: Authentication services

Machine Learning Algorithm Used : KNN (trained in Google Colab and saved in Google Drive)
Accuracy : 91.39%, Average F1 Score : 91.5%
