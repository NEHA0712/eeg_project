import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("best_eeg_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# App title
st.title("🧠 EEG Eye State Prediction App")

st.write("Enter EEG signal values to predict eye state.")

# Input fields (14 EEG channels)
AF3 = st.number_input("AF3", value=0.0)
F7 = st.number_input("F7", value=0.0)
F3 = st.number_input("F3", value=0.0)
FC5 = st.number_input("FC5", value=0.0)
T7 = st.number_input("T7", value=0.0)
P7 = st.number_input("P7", value=0.0)
O1 = st.number_input("O1", value=0.0)
O2 = st.number_input("O2", value=0.0)
P8 = st.number_input("P8", value=0.0)
T8 = st.number_input("T8", value=0.0)
FC6 = st.number_input("FC6", value=0.0)
F4 = st.number_input("F4", value=0.0)
F8 = st.number_input("F8", value=0.0)
AF4 = st.number_input("AF4", value=0.0)

# Predict button
if st.button("Predict Eye State"):

    input_data = np.array([[AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4]])
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.success("👁️ Eye Closed")
    else:
        st.success("👁️ Eye Open")
