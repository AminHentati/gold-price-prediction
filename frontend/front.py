import streamlit as st
import requests
import pandas as pd

# Set the FastAPI backend URL
BACKEND_URL = "http://localhost:8000"

# Streamlit app title
st.title("Gold Price Prediction")

# User input form
st.write("Enter the following gold price details:")

# Input fields for Open, High, Low, and Volume
open_price = st.number_input("Open Price", min_value=0.0, value=1900.5)
high_price = st.number_input("High Price", min_value=0.0, value=1920.3)
low_price = st.number_input("Low Price", min_value=0.0, value=1895.2)
volume = st.number_input("Volume", min_value=0, value=1200)

# Button to make a prediction
if st.button("Predict"):
    # Create JSON data to send to the FastAPI backend
    input_data = {
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Volume": volume
    }

    try:
        # Send a POST request to the FastAPI backend
        response = requests.post(f"{BACKEND_URL}/predict", json=input_data)
        response.raise_for_status()  # Raise an error for bad status codes

        # Get the prediction response
        prediction = response.json().get("prediction", "Prediction not found")

        st.write(f"Predicted Gold Price: {prediction}")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the FastAPI server: {e}")
