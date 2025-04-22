import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv('FIREBASE_CREDENTIALS_PATH'))
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
    })

# Crop type mapping dictionary (name to encoded value)
crop_mapping = {
    "Coffee": 0,
    "Garden Flowers": 1,
    "Groundnuts": 2,
    "Maize": 3,
    "Paddy": 4,
    "Potato": 5,
    "Pulse": 6,
    "Sugarcane": 7,
    "Wheat": 8
}

# Function to convert crop name to number
def get_crop_code(crop_name):
    if isinstance(crop_name, int) and 0 <= crop_name <= 8:
        return crop_name  # Already a valid code
    elif crop_name in crop_mapping:
        return crop_mapping[crop_name]
    else:
        print(f"Warning: '{crop_name}' not found in crop mapping. Using default (Maize:3)")
        return 3  # Default to Maize if not found

# Get crop input from user
print("Available crops:", ", ".join(crop_mapping.keys()))
crop_input = input("Enter crop name (or press Enter for 'Groundnuts'): ").strip()
if not crop_input:
    crop_input = "Groundnuts"  # Default crop

# Get other sensor values
try:
    temp = float(input("Enter temperature (°C) [28.5]: ") or "28.5")
    humidity = float(input("Enter humidity (%) [65.3]: ") or "65.3")
    soil_moisture = float(input("Enter soil moisture (%) [35.2]: ") or "35.2")
except ValueError:
    print("Invalid input. Using default values.")
    temp = 28.5
    humidity = 65.3
    soil_moisture = 35.2

# Convert crop name to number
crop_code = get_crop_code(crop_input)

# Create test data
test_data = {
    "CropType": crop_code,
    "temperature": temp,
    "Humidity": humidity,
    "SoilMoisture": soil_moisture
}

# Print what we're sending to Firebase
print("\nSending to Firebase:")
print(f"  Crop: {crop_input} (code: {crop_code})")
print(f"  Temperature: {temp}°C")
print(f"  Humidity: {humidity}%")
print(f"  Soil Moisture: {soil_moisture}%")

# Send data to Firebase
sensor_ref = db.reference("sensor")
sensor_ref.set(test_data)

# Store sensor data permanently in Realtime Database (instead of Firestore)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
record = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "sensor_data": test_data,
    "crop_name": crop_input,
    "crop_code": crop_code
}

# Add to the "history/sensors" node in Realtime Database
history_ref = db.reference("history/sensors")
history_ref.child(timestamp).set(record)

print("\nTest data added to Firebase successfully!")
print(f"Data permanently stored in database with ID: {timestamp}")
print("Run 1stcode.py to get the irrigation prediction.")