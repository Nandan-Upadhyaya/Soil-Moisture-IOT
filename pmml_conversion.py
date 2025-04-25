# Google Colab script to convert joblib models to PMML format

# Check scikit-learn version first and restart if needed
import sklearn
print(f"Current scikit-learn version: {sklearn.__version__}")

if sklearn.__version__ != "1.5.2":
    print("Downgrading scikit-learn to version 1.5.2 (required for sklearn2pmml)...")
    !pip install scikit-learn==1.5.2
    
    # Force a restart of the Colab runtime
    print("\nNEED TO RESTART RUNTIME for package changes to take effect!")
    print("Please run this cell, wait for downgrade to complete, then:")
    print("1. Click 'Runtime' in the menu")
    print("2. Click 'Restart runtime'")
    print("3. Run this notebook again\n")
    
    # This will only run if we're in Colab
    try:
        from google.colab import runtime
        runtime.unassign()  # This will restart the runtime
    except:
        print("Not running in Colab or couldn't restart automatically.")
        print("Please restart the runtime manually.")
    
    import sys
    sys.exit()  # Exit to force restart if automatic restart fails

# Continue if we have the right version
print("Using correct scikit-learn version for PMML conversion.")
!pip install sklearn2pmml

# Now load the model files and continue with conversion
import joblib
import pickle
import numpy as np
import os

# Load the existing files
model = joblib.load('YOUR MODEL PATH HERE')
scaler = joblib.load('YOUR SCALER PATH HERE')

# Load feature names
with open('/content/drive/MyDrive/KNNModel/feature_names.pkl', 'rb') as f:
    feature_names_dict = pickle.load(f)

# Print feature information
feature_list = feature_names_dict['features']
print(f"Features used by the model: {feature_list}")
n_features = len(feature_list)
print(f"Number of features: {n_features}")

# Create a simple test input for verification
test_input = np.random.rand(1, n_features).astype(np.float32)
print(f"Test input shape: {test_input.shape}")

# Test original model prediction
original_input = scaler.transform(test_input)
original_pred = model.predict(original_input)
print(f"Original model prediction: {original_pred}")

# Convert to PMML format (platform-independent)
print("\nConverting model to PMML format...")

# Make sure we're using the right scikit-learn version for sklearn2pmml
import sklearn
print(f"Using scikit-learn version: {sklearn.__version__}")

from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

# Create a pipeline that includes both the scaler and the model
pmml_pipeline = PMMLPipeline([
    ("scaler", scaler),
    ("model", model)
])

# Also create a separate pipeline for just the scaler
scaler_pipeline = PMMLPipeline([
    ("scaler", scaler)
])

# Configure both pipelines
pmml_pipeline.configure(compact=False, with_repr=True)
scaler_pipeline.configure(compact=False, with_repr=True)

# Set fields to avoid warnings
pmml_pipeline.active_fields = feature_list
pmml_pipeline.target_fields = ["y"]

scaler_pipeline.active_fields = feature_list
scaler_pipeline.target_fields = ["scaled_features"]

# Save the full pipeline to PMML format
pmml_path = '/content/drive/MyDrive/knn_irrigation_model.pmml'
scaler_pmml_path = '/content/drive/MyDrive/knn_scaler.pmml'

# Double check before running sklearn2pmml
if sklearn.__version__ != "1.5.2":
    print("ERROR: Wrong scikit-learn version! Must be 1.5.2 for compatibility.")
    print("Please restart the runtime and try again.")
    import sys
    sys.exit(1)

# Save both pipelines to PMML
print("Saving combined model+scaler pipeline to PMML...")
sklearn2pmml(pmml_pipeline, pmml_path, debug=True)
print(f"Full pipeline saved to {pmml_path}")

print("Saving standalone scaler to PMML...")
sklearn2pmml(scaler_pipeline, scaler_pmml_path, debug=True)
print(f"Scaler saved to {scaler_pmml_path}")

# Test the original pipeline with same test data to verify
pmml_pipeline_pred = pmml_pipeline.predict(test_input)
print(f"PMML Pipeline prediction: {pmml_pipeline_pred}")
print(f"Prediction match with original: {original_pred == pmml_pipeline_pred}")

print("\nPMML conversion complete! The PMML file can be used across different platforms.")
print("PMML is platform-independent using XML format.")
print("PMML can be used in many systems including Java, R, Python, and various scoring engines.")
print("It will work on your Raspberry Pi without architecture compatibility issues.")
