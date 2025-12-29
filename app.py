# app.py
import streamlit as st
import os
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="EEG Signal Prediction", layout="centered")

st.title("EEG Signal Classification App")

# Load model and scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_eeg_model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")

# Load model
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found! Make sure 'best_eeg_model.pkl' is in the folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load scaler
try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error("Scaler file not found! Make sure 'scaler.pkl' is in the folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

# Input fields
st.subheader("Enter EEG features")
# Example: if your model expects 5 features, adjust accordingly
feature_1 = st.number_input("Feature 1")
feature_2 = st.number_input("Feature 2")
feature_3 = st.number_input("Feature 3")
feature_4 = st.number_input("Feature 4")
feature_5 = st.number_input("Feature 5")

# Predict button
if st.button("Predict"):
    try:
        # Create numpy array of input features
        input_features = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5]])
        
        # Scale the input
        input_scaled = scaler.transform(input_features)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        st.success(f"Predicted class: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
