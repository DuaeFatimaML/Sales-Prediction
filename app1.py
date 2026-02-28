import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Sales Forecaster", layout="wide")

# --- 1. LOAD MODEL & FEATURES ---
@st.cache_resource
def load_assets():
    model = joblib.load('sales_model.joblib')
    features = joblib.load('features.joblib')
    return model, features

model, feature_names = load_assets()

# --- 2. USER INTERFACE ---
st.title("📊 Business Sales Prediction Dashboard")
st.write("Use this tool to predict sales for a specific date or generate a 30-day forecast.")

# Sidebar for Single Date Prediction
st.sidebar.header("Single Day Prediction")
pick_date = st.sidebar.date_input("Select Date")
# The user provides the TimeIndex manually (as it's a specific feature in your training)
time_idx = st.sidebar.number_input("Current Time Index", value=100)

if st.sidebar.button("Predict Sales"):
    # Create the data in the same format as the training
    input_dict = {
        'Year': pick_date.year,
        'Month': pick_date.month,
        'Day': pick_date.day,
        'TimeIndex': time_idx
    }
    input_df = pd.DataFrame([input_dict])
    
    # Ensure column order matches features.joblib
    input_df = input_df[feature_names]
    
    prediction = model.predict(input_df)
    st.sidebar.success(f"Estimated Sales: ${prediction[0]:,.2f}")

# --- 3. FUTURE FORECAST SECTION ---
st.header("🚀 30-Day Future Forecast")
if st.button("Generate Forecast Chart"):
    # Logical recreation of your training script's forecast
    future_dates = pd.date_range(start=pick_date, periods=30)
    
    future_df = pd.DataFrame({
        'Year': future_dates.year,
        'Month': future_dates.month,
        'Day': future_dates.day,
        'TimeIndex': np.arange(time_idx, time_idx + 30)
    })
    
    # Ensure column order
    future_df = future_df[feature_names]
    
    # Predict
    preds = model.predict(future_df)
    results = pd.DataFrame({'Date': future_dates, 'Predicted_Sales': preds})
    
    # Display Chart and Table
    col1, col2 = st.columns([2, 1])
    with col1:
        st.line_chart(results.set_index('Date'))
    with col2:
        st.write(results)