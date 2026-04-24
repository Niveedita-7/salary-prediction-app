import streamlit as st
import pickle
import numpy as np

# Load model & scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Title
st.title("Salary Prediction App")
st.write("Predict salary based on rating and number of salaries reported")

# Inputs (match dataset columns)
rating = st.number_input("Company Rating", min_value=0.0, max_value=5.0)
reports = st.number_input("Salaries Reported", min_value=0)

# Predict
if st.button("Predict Salary"):
    input_data = np.array([[rating, reports]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.subheader("Predicted Salary:")
    st.success(f"₹ {prediction[0]:,.2f}")