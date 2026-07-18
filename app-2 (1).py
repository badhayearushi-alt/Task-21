# ==========================================
# Q1: Import Required Libraries
# ==========================================

import streamlit as st
import pandas as pd
import joblib

# ==========================================
# Q2: Load Model and Preprocessing Files
# ==========================================

model = joblib.load("LR_model.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

# ==========================================
# Q3: Configure Streamlit Page
# ==========================================

st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)

# ==========================================
# Q4: Title and Description
# ==========================================

st.title("🚗 Ford Car Price Prediction")

st.write("Enter the car details below to predict the selling price.")

# ==========================================
# Q5: Numerical Input Fields
# ==========================================

year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2025,
    value=2020
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    value=10000
)

tax = st.number_input(
    "Road Tax",
    min_value=0,
    value=150
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    value=50.0
)

engineSize = st.number_input(
    "Engine Size",
    min_value=0.0,
    value=1.5
)

# ==========================================
# Q6: Dropdown Fields
# ==========================================

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic", "Semi-Auto"]
)

fuelType = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric"]
)

# ==========================================
# Q7: Text Input and Predict Button
# ==========================================

model_name = st.text_input("Car Model")

predict = st.button("Predict Price")

# ==========================================
# Q8 & Q9: Data Preparation and Prediction
# ==========================================

if predict:

    input_data = pd.DataFrame({
        "year":[year],
        "mileage":[mileage],
        "tax":[tax],
        "mpg":[mpg],
        "engineSize":[engineSize],
        "transmission":[transmission],
        "fuelType":[fuelType],
        "model":[model_name]
    })

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(f"Predicted Car Price: £{prediction[0]:,.2f}")

# ==========================================
# Q10: End of Application
# ==========================================
