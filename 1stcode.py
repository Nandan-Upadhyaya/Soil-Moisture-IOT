import firebase_admin
from firebase_admin import credentials, db
import joblib, pickle
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Only initialize if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv('FIREBASE_CREDENTIALS_PATH'))
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
    })

# Load model, scaler, feature names
model = joblib.load(os.getenv('MODEL_PATH'))
scaler = joblib.load(os.getenv('SCALER_PATH'))
with open(os.getenv('FEATURE_NAMES_PATH'), "rb") as f:
    feature_names_dict = pickle.load(f)
    
# Extract the list of feature names
feature_list = feature_names_dict['features']
print(f"Required features: {feature_list}")

# Get sensor data from Firebase
sensor_data = db.reference("sensor").get()
if not sensor_data:
    print("No sensor data found.")
else:
    print("Raw sensor data from Firebase:", sensor_data)
    
    # Create a properly formatted input array with all required features
    input_data = []
    for feature in feature_list:
        if feature in sensor_data:
            input_data.append(sensor_data[feature])
        else:
            print(f"Warning: Feature '{feature}' not found in sensor data. Using default value 0.")
            input_data.append(0)
    
    print(f"Processed input data: {input_data}")
    
    # Verify we have the correct number of features
    if len(input_data) != len(feature_list):
        print(f"Error: Input data has {len(input_data)} features, but model expects {len(feature_list)} features.")
    else:
        # Create a DataFrame with feature names to avoid the warning
        import pandas as pd
        input_df = pd.DataFrame([input_data], columns=feature_list)
        
        # Predict
        scaled = scaler.transform(input_df)
        prediction = int(model.predict(scaled)[0])

        # Push prediction back to Firebase
        db.reference("prediction").set(prediction)
        print(f"Prediction sent to Firebase: {prediction} ({'Irrigation needed' if prediction == 1 else 'No irrigation needed'})")
        
        # Store data permanently in Realtime Database instead of Firestore
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create data record with all relevant information
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sensor_data": sensor_data,
            "prediction": prediction,
            "prediction_text": 'Irrigation needed' if prediction == 1 else 'No irrigation needed'
        }
        
        # Add to the "history/predictions" node in Realtime Database
        history_ref = db.reference("history/predictions")
        history_ref.child(timestamp).set(record)
        print(f"Data permanently stored in database with ID: {timestamp}")
