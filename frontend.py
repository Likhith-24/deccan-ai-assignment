# Import necessary libraries
import streamlit as st              # For building the UI
import requests                     # For making API requests
import json                         # For JSON handling

# Streamlit app title
st.title("House Price Prediction")  # Sets the page title

# Input fields for house features
st.header("Enter House Features")   # Adds a section header
med_inc = st.number_input("Median Income", min_value=0.0, step=0.1)  # Input for MedInc
house_age = st.number_input("House Age", min_value=0.0, step=1.0)    # Input for HouseAge
ave_rooms = st.number_input("Average Rooms", min_value=0.0, step=0.1)  # Input for AveRooms
ave_bedrms = st.number_input("Average Bedrooms", min_value=0.0, step=0.1)  # Input for AveBedrms
population = st.number_input("Population", min_value=0.0, step=1.0)  # Input for Population
ave_occup = st.number_input("Average Occupancy", min_value=0.0, step=0.1)  # Input for AveOccup
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.01)  # Input for Latitude
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.01)  # Input for Longitude

# Button to make prediction
if st.button("Predict"):            # Creates a Predict button
    # Prepare data for API request
    data = {                        # Dictionary with input features
        "MedInc": med_inc,          # Median income from input
        "HouseAge": house_age,      # House age from input
        "AveRooms": ave_rooms,      # Average rooms from input
        "AveBedrms": ave_bedrms,    # Average bedrooms from input
        "Population": population,   # Population from input
        "AveOccup": ave_occup,      # Average occupancy from input
        "Latitude": latitude,       # Latitude from input
        "Longitude": longitude      # Longitude from input
    }

    # Send POST request to API
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=data)  # Sends request to local API
        response.raise_for_status()  # Raises exception for bad status codes (e.g., 400, 500)
        prediction = response.json()["prediction"]  # Extracts prediction from response
        st.success(f"Predicted House Price: ${prediction:.2f} (in $100,000s)")  # Displays success message
    except requests.exceptions.RequestException as e:  # Catches request errors
        st.error(f"Error: {str(e)}")  # Displays error message