# Read real-time sensor data from Arduino
import firebase_admin
from firebase_admin import credentials, db, firestore
import serial
import time
from datetime import datetime
import os
import numpy as np
import joblib
import pickle
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import io
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Firebase initialization with credentials from environment variables
if not firebase_admin._apps:
    try:
        # Get Firebase config from environment variables
        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
        db_url = os.getenv("FIREBASE_DATABASE_URL")
        
        # Check if environment variables are available
        if not cred_path or not db_url:
            print("WARNING: Firebase environment variables not found.")
            print("Create a .env file with FIREBASE_CREDENTIALS_PATH and FIREBASE_DATABASE_URL")
        
        # Use credentials from environment
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': db_url
        })
        print("Firebase initialized successfully")
    except Exception as e:
        print(f"Firebase initialization error: {e}")
        print("Continuing without Firebase - data will not be saved to the cloud")

# Initialize Firestore with error handling
try:
    db_firestore = firestore.client()
    print("Firestore client initialized")
except Exception as e:
    print(f"Firestore initialization error: {e}")
    db_firestore = None

# Helper function to safely store data in Firebase
def safe_firebase_store(sensor_data, prediction):
    """Store data in Firebase Realtime Database with permanent history"""
    
    try:
        # 1. Store current reading (for current state)
        db.reference("current/sensor").set(sensor_data)
        db.reference("current/prediction").set(prediction)
        
        # 2. Also store in a history node with timestamp (permanent record)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        history_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sensor_data": sensor_data,
            "prediction": prediction,
            "prediction_text": 'Irrigation needed' if prediction == 1 else 'No irrigation needed'
        }
        db.reference(f"history/{timestamp}").set(history_record)
        print("Data sent to Firebase Realtime Database (current state and permanent history)")
    except Exception as e:
        if "404" in str(e):
            print("Firebase Realtime DB error: Database not found")
            print("Fix: Create a Realtime Database in your Firebase console")
        elif "403" in str(e):
            print("Firebase Realtime DB error: Permission denied")
            print("Fix: Check your database rules")
        else:
            print(f"Firebase Realtime DB error: {e}")
    
    # Skip Firestore storage completely

# Google Drive authentication for Raspberry Pi
def authenticate_google_drive():
    try:
        print("Authenticating with Google Drive...")
        gauth = GoogleAuth()
        
        # Set the path to client_secrets.json from environment variables
        client_secrets_path = os.getenv("GOOGLE_CLIENT_SECRETS_PATH")
        gauth.settings['client_config_file'] = client_secrets_path
        
        if not os.path.exists(client_secrets_path):
            print(f"ERROR: client_secrets.json not found at {client_secrets_path}")
            print("Please make sure you've downloaded it from Google Cloud Console")
            return None
        
        # Define specific scopes for our use case from environment variables
        gauth.settings['scope'] = [os.getenv("GOOGLE_DRIVE_SCOPE", "https://www.googleapis.com/auth/drive.file")]
        
        # Try to load saved client credentials
        gauth.LoadCredentialsFile("mycreds.txt")
        if gauth.credentials is None:
            # Authenticate if they're not there
            print("\n=== IMPORTANT AUTHENTICATION INSTRUCTIONS ===")
            print("ERROR: You're seeing an access_denied error because:")
            print("1. Your email is not added as a test user in Google Cloud Console")
            print("2. Go to Google Cloud Console > APIs & Services > OAuth consent screen")
            print("3. Add your email (the one you're trying to log in with) as a test user")
            print("4. You may also need to enable the Drive API in Google Cloud Console")
            print("\nWould you like to:")
            print("1. Try authentication with a different method")
            print("2. Exit and fix the test user settings")
            choice = input("Enter your choice (1/2): ")
            
            if choice == '1':
                print("Trying with command-line authentication...")
                try:
                    gauth.CommandLineAuth()  # This uses command-line authentication instead
                except Exception as cmd_err:
                    print(f"Command-line authentication failed: {cmd_err}")
                    print("Creating a simplified settings file for direct API access...")
                    
                    # Create a simplified authentication method
                    print("\nFOLLOW THESE STEPS:")
                    print("1. Go to https://console.cloud.google.com/apis/credentials")
                    print("2. Create an API Key (not OAuth)")
                    print("3. Copy the API key")
                    api_key = input("Enter the API key: ").strip()
                    
                    # Write a simple configuration file
                    with open("drive_api_key.txt", "w") as f:
                        f.write(api_key)
                    
                    print("API key saved. Please restart the application.")
                    return None
            else:
                print("Exiting. Please add your email as a test user and try again.")
                return None
        elif gauth.access_token_expired:
            # Refresh them if expired
            gauth.Refresh()
        else:
            # Initialize the saved creds
            gauth.Authorize()
            
        # Save the current credentials
        gauth.SaveCredentialsFile("mycreds.txt")
        drive = GoogleDrive(gauth)
        print("Google Drive authenticated successfully")
        return drive
    except Exception as e:
        print(f"Google Drive authentication error: {e}")
        print("\nTROUBLESHOOTING STEPS:")
        print("1. Ensure your email is added as a test user")
        print("2. Verify you have enabled the Google Drive API")
        print("3. Check that your client_secrets.json is correct")
        print("4. Try creating a new OAuth 2.0 Client ID in Google Cloud Console")
        return None

# Access ML models directly in Google Drive (no local download)
def access_ml_models(drive):
    model_session, feature_names_dict = None, None
    try:
        print("Accessing ML models directly from Google Drive...")
        
        # Check for Java installation first (required by pypmml)
        java_installed = False
        try:
            import subprocess
            result = subprocess.run(['java', '-version'], capture_output=True, text=True)
            if result.returncode == 0:
                java_installed = True
                print("Java found, continuing with pypmml.")
            else:
                print("Java not found. PMML loading requires Java.")
                print("Please install Java with: 'sudo apt-get install default-jre'")
        except FileNotFoundError:
            print("Java not found. PMML loading requires Java.")
            print("Please install Java with: 'sudo apt-get install default-jre'")
        
        if not java_installed:
            print("Attempting to install Java automatically...")
            try:
                # Try to install Java
                install_result = subprocess.run(
                    ['sudo', 'apt-get', 'update'], 
                    capture_output=True, text=True
                )
                install_result = subprocess.run(
                    ['sudo', 'apt-get', '-y', 'install', 'default-jre'], 
                    capture_output=True, text=True
                )
                
                if install_result.returncode == 0:
                    print("Java installed successfully!")
                    java_installed = True
                else:
                    print("Failed to install Java automatically.")
                    print("Error details:", install_result.stderr)
                    # Fall back to direct joblib loading option
                    return load_models_direct(drive)
            except Exception as e:
                print(f"Error during Java installation: {e}")
                # Fall back to direct joblib loading option
                return load_models_direct(drive)
        
        # Check/install pypmml if needed
        try:
            import importlib.util
            if importlib.util.find_spec("pypmml") is None:
                print("pypmml not found, attempting to install...")
                import subprocess
                install_result = subprocess.run(
                    ["pip", "install", "pypmml"],
                    capture_output=True, text=True
                )
                
                if install_result.returncode != 0:
                    print("Error installing pypmml via pip.")
                    print("Error details:", install_result.stderr)
                    raise ImportError("Failed to install pypmml")
                else:
                    print("pypmml installed successfully")
            import pypmml
            print(f"pypmml imported successfully")
        except Exception as e:
            print(f"Error with pypmml: {e}")
            print("Cannot continue without pypmml")
            return None, None
        
        # File IDs for your ML model files in Google Drive from environment variables
        model_file_id = os.getenv("MODEL_FILE_ID")
        features_file_id = os.getenv("FEATURES_FILE_ID")
        
        # Use temporary files to handle binary data
        temp_dir = tempfile.mkdtemp()
        
        # Feature names - still using pickle for feature names
        print("Loading feature names...")
        features_temp_path = os.path.join(temp_dir, "temp_features.pkl")
        features_file = drive.CreateFile({'id': features_file_id})
        features_file.GetContentFile(features_temp_path)
        with open(features_temp_path, "rb") as f:
            feature_names_dict = pickle.load(f)
        
        # Download PMML model to temp file then load
        print("Loading PMML model...")
        model_temp_path = os.path.join(temp_dir, "temp_model.pmml")
        model_file = drive.CreateFile({'id': model_file_id})
        model_file.GetContentFile(model_temp_path)
        
        # Load the PMML model
        model_session = pypmml.Model.load(model_temp_path)
        print("PMML model loaded successfully")
        
        print("ML model components accessed successfully from cloud")
        
        # Clean up temp files (optional)
        for file_path in [model_temp_path, features_temp_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)
        
    except Exception as e:
        print(f"Error accessing PMML models from Drive: {e}")
        # Try fallback method with direct joblib loading
        return load_models_direct(drive)
    
    return model_session, feature_names_dict

# Fallback method - load joblib files directly
def load_models_direct(drive):
    try:
        print("\nFalling back to direct loading of joblib files...")
        
        # File IDs for your joblib model files - replace with actual IDs
        model_file_id = "YOUR_JOBLIB_MODEL_FILE_ID"  # original joblib file
        scaler_file_id = "YOUR_JOBLIB_SCALER_FILE_ID"  # original joblib file  
        features_file_id = "1--LhlHikjYTh4FYDLpHW64YH7z4LmP3h"  # should work as is
        
        # Use temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Feature names
        print("Loading feature names...")
        features_temp_path = os.path.join(temp_dir, "temp_features.pkl")
        features_file = drive.CreateFile({'id': features_file_id})
        features_file.GetContentFile(features_temp_path)
        with open(features_temp_path, "rb") as f:
            feature_names_dict = pickle.load(f)
        
        # Create a wrapper class that mimics PMML model interface
        class JobLibModelWrapper:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
                
            def predict(self, features_dict):
                # Convert dict to array in correct order
                features_array = []
                for feat in feature_names_dict['features']:
                    features_array.append(features_dict[feat])
                
                # Scale and predict
                scaled_features = self.scaler.transform([features_array])
                prediction = self.model.predict(scaled_features)[0]
                return {'prediction': int(prediction)}
        
        # Try loading model and scaler
        try:
            print("Please provide the file IDs for your original joblib files in Google Drive.")
            print("You can find these in your Google Drive URL when viewing the file.")
            
            model_file_id = input("Enter model joblib file ID: ").strip() or model_file_id
            scaler_file_id = input("Enter scaler joblib file ID: ").strip() or scaler_file_id
            
            # Download and load model
            print("Loading model joblib...")
            model_temp_path = os.path.join(temp_dir, "model.joblib")
            model_file = drive.CreateFile({'id': model_file_id})
            model_file.GetContentFile(model_temp_path)
            model = joblib.load(model_temp_path)
            
            # Download and load scaler
            print("Loading scaler joblib...")
            scaler_temp_path = os.path.join(temp_dir, "scaler.joblib")
            scaler_file = drive.CreateFile({'id': scaler_file_id})
            scaler_file.GetContentFile(scaler_temp_path)
            scaler = joblib.load(scaler_temp_path)
            
            # Create wrapper model
            wrapped_model = JobLibModelWrapper(model, scaler)
            
            print("Successfully loaded model files directly")
            
            # Clean up temp files
            for file_path in [model_temp_path, scaler_temp_path, features_temp_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            os.rmdir(temp_dir)
            
            return wrapped_model, feature_names_dict
            
        except Exception as je:
            print(f"Error loading joblib files: {je}")
            return None, None
            
    except Exception as e:
        print(f"Error in direct loading fallback: {e}")
        return None, None

# PMML prediction function
def predict_with_pmml(pmml_model, input_data, feature_list):
    try:
        # Convert input data to dictionary as required by PMML
        features = {}
        for i, feature in enumerate(feature_list):
            features[feature] = float(input_data[i])
        
        # Make prediction with PMML model
        result = pmml_model.predict(features)
        
        # Debug output to understand the model result structure
        result_type = type(result).__name__
        print(f"PMML model returned: {result_type}")
        
        # Handle JavaMap specially (from pypmml)
        if result_type == "JavaMap":
            print("Processing JavaMap result")
            # For KNN models, extract target values from neighbors and take majority vote
            neighbor_values = []
            for key in result:
                if key.startswith('neighbor('):
                    try:
                        # Get the value as a string and convert to int
                        value_str = str(result[key])
                        neighbor_values.append(int(value_str))
                    except ValueError:
                        # If not numeric, skip this value
                        print(f"Warning: Skipping non-numeric neighbor value: {key}={result[key]}")
            
            if neighbor_values:
                # Count the occurrences of each value
                from collections import Counter
                value_counts = Counter(neighbor_values)
                
                # Find the most common value (the mode)
                most_common = value_counts.most_common(1)[0][0]
                print(f"Most common neighbor value: {most_common}")
                print(f"All neighbor values: {neighbor_values[:5]}... (total: {len(neighbor_values)})")
                
                # Use a threshold to determine irrigation needed (0 or 1)
                if most_common > 150:
                    print("Threshold > 150, irrigation needed")
                    return 1  # Irrigation needed
                else:
                    print("Threshold <= 150, no irrigation needed")
                    return 0  # No irrigation needed
            else:
                print("No valid neighbor values found")
                return 0
                
        # Extract prediction for Python dict or other types
        # ...existing code...
        
    except Exception as e:
        print(f"Error during PMML prediction: {e}")
        print(f"Input features: {features}")
        
        # Special handling for JavaMap in exception case
        if 'result' in locals() and type(result).__name__ == "JavaMap":
            try:
                print("Attempting recovery from JavaMap error...")
                neighbor_values = []
                for key in result:
                    if key.startswith('neighbor('):
                        try:
                            neighbor_values.append(int(str(result[key])))
                        except (ValueError, TypeError):
                            pass
                
                if neighbor_values:
                    from collections import Counter
                    value_counts = Counter(neighbor_values)
                    most_common = value_counts.most_common(1)[0][0]
                    print(f"Recovered most common value: {most_common}")
                    
                    if most_common > 150:
                        print("Recovery decision: Irrigation needed")
                        return 1
                    else:
                        print("Recovery decision: No irrigation needed")
                        return 0
            except Exception as inner_e:
                print(f"Error in JavaMap recovery: {inner_e}")
        
        # Default to no irrigation when errors occur
        print("Due to prediction error, defaulting to 'No irrigation needed'")
        return 0

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

# Authenticate with Google Drive
drive_client = authenticate_google_drive()
if not drive_client:
    print("Failed to authenticate with Google Drive. Cannot continue.")
    exit(1)

# Ask for the serial port
print("Please connect your Arduino to a USB port.")
port = input("Enter Arduino serial port (e.g., /dev/ttyACM0, COM3): ") or "/dev/ttyACM0"

# Get crop input from user
print("Available crops:", ", ".join(crop_mapping.keys()))
crop_input = input("Enter crop name (or press Enter for 'Groundnuts'): ").strip()
if not crop_input:
    crop_input = "Groundnuts"  # Default crop
crop_code = get_crop_code(crop_input)
print(f"Selected crop: {crop_input} (code: {crop_code})")

# Access models directly from Google Drive (no local download)
try:
    # Using PyDrive to access models in Google Drive (no local download)
    model, feature_names_dict = access_ml_models(drive_client)
    
    # Extract feature list from the loaded feature names
    feature_list = feature_names_dict['features']
    print(f"Required features: {feature_list}")
except Exception as e:
    print(f"Failed to load models: {e}")
    exit(1)

if model is None or feature_names_dict is None:
    print("Failed to access ML models. Cannot continue with predictions.")
    exit(1)

# How many readings to collect
max_readings = int(input("Enter number of readings to collect (or press Enter for continuous): ") or "0")
readings = 0

print("\nReading data from Arduino sensors...")
print("Press Ctrl+C to stop")

# Connect to Arduino
try:
    ser = serial.Serial(port, 9600, timeout=1)
    ser.flush()
    print(f"Connected to Arduino on {port}")

    # Initialize sensor data dictionary with correct case for temperature
    sensor_data = {
        "CropType": crop_code,
        "SoilMoisture": 0,
        "temperature": 25,  # Changed to lowercase to match feature_list
        "Humidity": 50
    }

    try:
        while max_readings == 0 or readings < max_readings:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').rstrip()

                # Skip error messages or unnecessary lines
                if line.startswith("ERROR:") or not line.strip():
                    print(f"Arduino error or empty: {line}")
                    continue

                print(f"Raw data from Arduino: {line}")

                # Parse soil moisture and other sensor data from Arduino
                if "Soil Moisture Value" in line:
                    try:
                        value_part = line.split(':')[1].strip()  # e.g., "526 (48%)"
                        moisture_value = int(value_part.split(' ')[0])  # Extract just the number: 526
                        print(f"Received Soil Moisture Value: {moisture_value}")
                        sensor_data["SoilMoisture"] = moisture_value
                        
                        # Process all available sensor data and make prediction
                        # Create input data array for the model
                        input_data = []
                        for feature in feature_list:
                            # Handle case sensitivity in feature names
                            feature_upper = feature.capitalize()
                            if feature in sensor_data:
                                input_data.append(sensor_data[feature])
                            elif feature_upper in sensor_data:
                                # If lowercase feature name not found, try capitalized version
                                print(f"Using {feature_upper} instead of {feature}")
                                input_data.append(sensor_data[feature_upper])
                            else:
                                print(f"Warning: Feature '{feature}' not found in sensor data. Using default value 0.")
                                input_data.append(0)
                        
                        # Make prediction using PMML model
                        if len(input_data) == len(feature_list):
                            prediction = predict_with_pmml(model, input_data, feature_list)
                            
                            if prediction is not None:
                                prediction_text = 'Irrigation needed' if prediction == 1 else 'No irrigation needed'
                                print(f"Prediction: {prediction_text} ({prediction})")
                                
                                # Send prediction back to Arduino
                                ser.write(f"PREDICTION:{prediction}\n".encode())
                                
                                # Store data in Firebase (with error handling)
                                safe_firebase_store(sensor_data, prediction)
                        
                        readings += 1
                        if max_readings > 0:
                            print(f"Readings collected: {readings}/{max_readings}")
                    except Exception as e:
                        print(f"Failed to parse soil moisture value: {e}")
                
                # Parse other sensor data if available (temperature, humidity, etc.)
                elif "Temperature" in line:
                    try:
                        temp_value = float(line.split(':')[1].strip())
                        # Store with lowercase key to match feature list
                        sensor_data["temperature"] = temp_value  
                        print(f"Temperature updated: {temp_value} (stored as 'temperature')")
                    except Exception as e:
                        print(f"Failed to parse temperature: {e}")
                
                elif "Humidity" in line:
                    try:
                        humidity_value = float(line.split(':')[1].strip())
                        sensor_data["Humidity"] = humidity_value
                        print(f"Humidity updated: {humidity_value}")
                    except Exception as e:
                        print(f"Failed to parse humidity: {e}")

            time.sleep(0.1)  # Small delay to prevent CPU overuse
    except KeyboardInterrupt:
        print("\nData collection stopped by user")

except serial.SerialException as e:
    print(f"Serial connection error: {e}")

finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial connection closed")

