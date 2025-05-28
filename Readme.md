SmartSoil, a comprehensive IoT-based smart irrigation management system that leverages machine learning to optimize water usage in agriculture. This project seamlessly integrates hardware components, cloud services, and AI to create an end-to-end smart farming solution.
<br>
<h1>🌱 Project Overview </h1>
Designed and implemented an automated irrigation system that analyzes real-time soil moisture, temperature, and humidity data to make intelligent irrigation decisions tailored to specific crop requirements (supporting Coffee, Garden Flowers, Groundnuts, Maize, Paddy, Potato, Pulse, Sugarcane, and Wheat). Our system also allows the user to input the crop name and then based on the crop and the sensor data associated with that crop, predictions are made and processed accordingly.
<br> 

<h1> Technical Architecture </h1> 
Hardware Layer: Arduino microcontroller interfaced with soil moisture sensors, temperature/humidity sensors (DHT11), and servo motors for physical irrigation control <br>
Processing Layer: Raspberry Pi running Python for data processing, ML inference, and system orchestration <br>
Cloud Integration: Real-time and historical data storage using Firebase Realtime Database <br>
ML Pipeline: K-Nearest Neighbors (KNN) classifier deployed as PMML model, stored on Google Drive and dynamically loaded to the edge device (Google Drive authentication using Pydrive with OAuth Credentials) <br>

<br>

<h1> Additional Libraries used: </h1> <br>
firebase-admin: Firebase data storage <br>
pydrive: Google Drive API integration <br>
pypmml: Processing PMML model files <br>
pyserial: Serial communication with Arduino <br>
pickle: Object serialization <br>
DHT11 sensor library <br>
Servo library <br>

<h1> Cloud Services: </h1> 
Firebase Realtime Database: Data storage & retrieval <br>
Google Drive API: Remote model storage <br>
Google Cloud OAuth: Authentication services <br>

Machine Learning Algorithm Used : KNN (trained in Google Colab and saved in Google Drive) <br>
Accuracy : 91.39%, Average F1 Score : 91.5% <br>
