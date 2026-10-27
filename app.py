import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------------
# Load saved model + encoders
# (These files must be generated from the notebook — see the
#  extra cell shared alongside this app: saves
#  'linear_regression_model.pkl', 'encoders.pkl', 'feature_order.pkl')
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("linear_regression_model.pkl")
    encoders = joblib.load("encoders.pkl")          # dict: {col_name: fitted LabelEncoder}
    feature_order = joblib.load("feature_order.pkl")  # list of column names in training order
    return model, encoders, feature_order

model, encoders, feature_order = load_artifacts()

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Used Car Price Predictor")
st.write("Linear Regression model trained on used car sales data.")

# ---------------------------------------------------------------
# Build input widgets
# - categorical columns -> dropdown using classes_ from the saved encoder
# - numeric columns -> number_input
# ---------------------------------------------------------------
st.subheader("Enter Car Details")

user_input = {}
categorical_cols = list(encoders.keys())

for col in feature_order:
    if col in categorical_cols:
        options = list(encoders[col].classes_)
        user_input[col] = st.selectbox(col, options)
    elif col == "Year":
        user_input[col] = st.number_input(col, min_value=1960, max_value=2026, value=2015, step=1)
    elif col == "Mileage":
        user_input[col] = st.number_input(col, min_value=0.0, value=100.0, step=1.0)
    elif col == "EngineV":
        user_input[col] = st.number_input(col, min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    else:
        user_input[col] = st.number_input(col, value=0.0)

if st.button("Predict Price"):
    # Build a single-row dataframe in the correct column order
    row = {}
    for col in feature_order:
        val = user_input[col]
        if col in categorical_cols:
            val = encoders[col].transform([val])[0]
        row[col] = val

    X_new = pd.DataFrame([row], columns=feature_order)

    log_price_pred = model.predict(X_new)[0]
    price_pred = np.exp(log_price_pred)   # model was trained on log(Price)

    st.success(f"Estimated Price: **${price_pred:,.2f}**")
