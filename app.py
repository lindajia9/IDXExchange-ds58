import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

rf_model = load_model()

# Get the exact feature names used during training.
# This lets the app recreate the one-hot encoded columns.
model_features = list(rf_model.feature_names_in_)

city_cols = [c for c in model_features if c.startswith("City_")]
county_cols = [c for c in model_features if c.startswith("CountyOrParish_")]
district_cols = [c for c in model_features if c.startswith("DistrictName_")]

# Remove the prefix to get the category names shown to the user.
city_options = [c.replace("City_", "", 1) for c in city_cols]
county_options = [c.replace("CountyOrParish_", "", 1) for c in county_cols]
district_options = [c.replace("DistrictName_", "", 1) for c in district_cols]

# -----------------------------
# App title
# -----------------------------
st.title("🏠 California House Price Predictor")
st.write(
    "Enter the property information below to predict the closing price "
    "using the trained Random Forest model."
)

# -----------------------------
# User inputs
# -----------------------------
st.subheader("Property Information")

living_area = st.number_input(
    "Living Area (sq ft)",
    min_value=0.0,
    value=1500.0,
    step=50.0
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=0,
    value=3,
    step=1
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=0.0,
    value=2.0,
    step=0.5
)

lot_size = st.number_input(
    "Lot Size",
    min_value=0.0,
    value=5000.0,
    step=500.0
)

age = st.number_input(
    "Age of Property (years)",
    min_value=0.0,
    value=20.0,
    step=1.0
)

latitude = st.number_input(
    "Latitude",
    value=34.0,
    format="%.6f"
)

longitude = st.number_input(
    "Longitude",
    value=-117.9,
    format="%.6f"
)

# Only display categorical selectors if those columns exist in the model.
city = st.selectbox("City", ["Unknown"] + city_options)
county = st.selectbox("County/Parish", ["Unknown"] + county_options)
district = st.selectbox("School District", ["Unknown"] + district_options)

# -----------------------------
# Make prediction
# -----------------------------
if st.button("Predict House Price", type="primary"):

    # Start with all model features set to zero.
    # This guarantees that the prediction dataframe has exactly
    # the same columns and order as the training data.
    input_data = pd.DataFrame(
        np.zeros((1, len(model_features))),
        columns=model_features
    )

    # Numerical features
    input_data.loc[0, "LivingArea"] = living_area
    input_data.loc[0, "BedroomsTotal"] = bedrooms
    input_data.loc[0, "BathroomsTotalInteger"] = bathrooms
    input_data.loc[0, "LotSizeArea"] = lot_size
    input_data.loc[0, "Age"] = age
    input_data.loc[0, "Latitude"] = latitude
    input_data.loc[0, "Longitude"] = longitude

    # One-hot encoded categorical features
    if city != "Unknown" and f"City_{city}" in input_data.columns:
        input_data.loc[0, f"City_{city}"] = 1

    if county != "Unknown" and f"CountyOrParish_{county}" in input_data.columns:
        input_data.loc[0, f"CountyOrParish_{county}"] = 1

    if district != "Unknown" and f"DistrictName_{district}" in input_data.columns:
        input_data.loc[0, f"DistrictName_{district}"] = 1

    # Model predicts log1p(ClosePrice), so convert prediction
    # back to the original dollar scale.
    predicted_log_price = rf_model.predict(input_data)[0]
    predicted_price = np.expm1(predicted_log_price)

    st.success(
        f"### Predicted Closing Price: ${predicted_price:,.0f}"
    )

    st.caption(
        "The Random Forest model was trained using log-transformed closing prices."
    )

# -----------------------------
# Sidebar information
# -----------------------------
with st.sidebar:
    st.header("About the Model")
    st.write("Model: Random Forest Regressor")
    st.write("Target: ClosePrice")
    st.write("Target transformation: log1p")
    st.write("Prediction converted back using expm1")
