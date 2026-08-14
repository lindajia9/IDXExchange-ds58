import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Streamlit app for the Random Forest model from the notebook
# ---------------------------------------------------------

st.set_page_config(
    page_title="California Home Price Predictor",
    page_icon="🏠",
    layout="centered"
)

MODEL_PATH = Path("model1.pkl")

st.title("🏠 California Home Price Predictor")
st.write(
    "Enter the property information below to estimate the closing price "
    "using the Random Forest model trained in the notebook."
)

# Load the trained Random Forest model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

rf_model = load_model()

if rf_model is None:
    st.error(
        "model1.pkl was not found. Put the trained Random Forest model "
        "file (model1.pkl) in the same GitHub repository as app.py."
    )
    st.stop()

# The notebook trained the model on a pandas DataFrame.
# RandomForestRegressor stores the feature names when fit with a DataFrame.
if not hasattr(rf_model, "feature_names_in_"):
    st.error(
        "The saved Random Forest model does not contain feature names. "
        "Please save the model after fitting it with X_train as a pandas DataFrame."
    )
    st.stop()

expected_features = list(rf_model.feature_names_in_)

st.subheader("Property information")

living_area = st.number_input(
    "Living Area (sq ft)",
    min_value=1.0,
    value=2000.0,
    step=100.0
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    value=3,
    step=1
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1.0,
    value=2.0,
    step=0.5
)

lot_size = st.number_input(
    "Lot Size Area",
    min_value=1.0,
    value=7000.0,
    step=500.0
)

year_built = st.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2000,
    step=1
)

latitude = st.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=33.95,
    format="%.6f"
)

longitude = st.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=-117.40,
    format="%.6f"
)

st.subheader("Location information")

city = st.text_input(
    "City",
    value="Riverside"
)

county = st.text_input(
    "County",
    value="Riverside"
)

district = st.text_input(
    "Unified School District",
    value="Unknown"
)

if st.button("Predict Close Price", type="primary"):

    # -----------------------------------------------------
    # Recreate the model input structure used in the notebook
    # -----------------------------------------------------

    # The notebook calculated Age as 2026 - YearBuilt.
    age = 2026 - year_built

    # Start with all expected model features set to zero.
    # This is important because the model contains one-hot encoded
    # City, CountyOrParish, and DistrictName columns.
    input_df = pd.DataFrame(
        np.zeros((1, len(expected_features))),
        columns=expected_features
    )

    # Numeric model features
    numeric_values = {
        "LivingArea": living_area,
        "BedroomsTotal": bedrooms,
        "BathroomsTotalInteger": bathrooms,
        "LotSizeArea": lot_size,
        "Age": age,
        "Latitude": latitude,
        "Longitude": longitude,
    }

    for feature, value in numeric_values.items():
        if feature in input_df.columns:
            input_df.at[0, feature] = value

    # -----------------------------------------------------
    # One-hot encoded categorical variables
    # -----------------------------------------------------
    #
    # The notebook used:
    #   pd.get_dummies(..., drop_first=True)
    #
    # Therefore, the reference category is represented by all
    # zeros. If a matching dummy column exists in the model,
    # turn that column on.
    #
    # City was first grouped so cities with fewer than 50
    # training observations became "Other".
    # -----------------------------------------------------

    city_dummy = f"City_{city}"
    county_dummy = f"CountyOrParish_{county}"
    district_dummy = f"DistrictName_{district}"

    if city_dummy in input_df.columns:
        input_df.at[0, city_dummy] = 1

    if county_dummy in input_df.columns:
        input_df.at[0, county_dummy] = 1

    if district_dummy in input_df.columns:
        input_df.at[0, district_dummy] = 1

    # Make absolutely sure the columns are in exactly the same
    # order as when the Random Forest was trained.
    input_df = input_df[expected_features]

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    # The notebook trained Random Forest on log1p(ClosePrice):
    #     y_train_log = np.log1p(y_train)
    #     rf_model.fit(X_train, y_train_log)
    #
    # Therefore, convert the prediction back to dollars with expm1.
    prediction_log = rf_model.predict(input_df)[0]
    predicted_price = np.expm1(prediction_log)

    st.success(
        f"### Estimated Close Price: ${predicted_price:,.0f}"
    )

    st.caption(
        "The Random Forest was trained on log-transformed ClosePrice, "
        "so the prediction shown above has been converted back to dollars."
    )

    # Optional display of the model inputs for transparency
    with st.expander("Show model inputs"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))
