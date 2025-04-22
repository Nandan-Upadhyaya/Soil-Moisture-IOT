import firebase_admin
from firebase_admin import credentials, db
import time
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Firebase (if not already initialized)
if not firebase_admin._apps:
    # Use credentials path from environment variables
    cred = credentials.Certificate(os.getenv('FIREBASE_CREDENTIALS_PATH'))
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
    })

def simulate_hardware():
    print("=== Smart Irrigation Hardware Simulator ===")
    print("Starting simulation...")

    # Create reference to the prediction node
    prediction_ref = db.reference("prediction")

    # Track servo state
    servo_on = False

    try:
        while True:
            # Get current prediction value
            prediction = prediction_ref.get()
            current_time = datetime.now().strftime("%H:%M:%S")

            print(f"\n[{current_time}] Checking prediction value...")
            print(f"Current prediction from Firebase: {prediction}")

            # Logic to determine if servo should be on/off
            # Assuming prediction = 1 means "turn on irrigation"
            if prediction == 1 and not servo_on:
                servo_on = True
                print("🔄 ACTION: Servo motor activated - IRRIGATION STARTED")
                # In real hardware, this would trigger GPIO pins to control the servo
            elif prediction == 0 and servo_on:
                servo_on = False
                print("🔄 ACTION: Servo motor deactivated - IRRIGATION STOPPED")
            else:
                print(f"🔄 Servo status: {'ON' if servo_on else 'OFF'} (No change needed)")

            # Hardware simulator status
            print(f"💧 Irrigation system status: {'ACTIVE' if servo_on else 'INACTIVE'}")

            # Wait before checking again
            print("Waiting 5 seconds before next check...")
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
        print("Simulation ended")

if __name__ == "__main__":
    simulate_hardware()