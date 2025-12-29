import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# -----------------------------
# Load model and scaler safely
# -----------------------------
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "best_eeg_model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

# -----------------------------
# Streamlit App Title
# -----------------------------
st.title("🧠 EEG Eye State Prediction App")
st.write("Predict if eyes are open or closed using EEG signals.")

# -----------------------------
# Sidebar for manual input
# -----------------------------
st.sidebar.header("Manual Input")
AF3 = st.sidebar.number_input("AF3")
F7 = st.sidebar.number_input("F7")
F3 = st.sidebar.number_input("F3")
FC5 = st.sidebar.number_input("FC5")
T7 = st.sidebar.number_input("T7")
P7 = st.sidebar.number_input("P7")
O1 = st.sidebar.number_input("O1")
O2 = st.sidebar.number_input("O2")
P8 = st.sidebar.number_input("P8")
T8 = st.sidebar.number_input("T8")
FC6 = st.sidebar.number_input("FC6")
F4 = st.sidebar.number_input("F4")
F8 = st.sidebar.number_input("F8")
AF4 = st.sidebar.number_input("AF4")

manual_input = np.array([[AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4]])

# -----------------------------
# CSV Upload Option
# -----------------------------
st.header("Or Upload CSV File")
uploaded_file = st.file_uploader("Upload a CSV file with 14 EEG columns", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(data)

    # Scale the data
    scaled_data = scaler.transform(data)
    predictions = model.predict(scaled_data)
    data['Predicted Eye State'] = ['Closed' if i==1 else 'Open' for i in predictions]

    st.write("Predictions:")
    st.dataframe(data)

# -----------------------------
# Manual Prediction Button
# -----------------------------
if st.button("Predict Manual Input"):
    scaled_manual = scaler.transform(manual_input)
    prediction = model.predict(scaled_manual)
    if prediction[0] == 1:
        st.success("👁️ Eye Closed")
    else:
        st.success("👁️ Eye Open")
