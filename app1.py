
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
 
# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Sales Predictor",
    page_icon="🛒",
    layout="centered"
)
 
st.title("🛒 Sales Prediction App")
st.markdown("Enter the feature values below to predict sales using the trained model.")
 
# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    model = joblib.load("model1.joblib")
    features = joblib.load("features1.joblib")
    return model, features
 
try:
    model, features = load_model()
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ model1.joblib or features1.joblib not found. Please place them in the same folder as app.py.")
    st.stop()
 
# ===================== FEATURE INPUTS =====================
st.subheader("📋 Input Features")
st.markdown("Fill in the values for each feature:")
 
input_data = {}
 
# Date-derived features — handled via a date picker
date_features = {"Year", "Month", "Day", "TimeIndex"}
 
if any(f in features for f in ["Year", "Month", "Day"]):
    st.markdown("#### 📅 Date")
    selected_date = st.date_input("Select Date", value=date.today())
 
    if "Year" in features:
        input_data["Year"] = selected_date.year
    if "Month" in features:
        input_data["Month"] = selected_date.month
    if "Day" in features:
        input_data["Day"] = selected_date.day
 
if "TimeIndex" in features:
    st.markdown("#### 🔢 Time Index")
    input_data["TimeIndex"] = st.number_input(
        "TimeIndex (sequential row number)",
        min_value=0,
        value=100,
        step=1
    )
 
# All remaining numeric features
other_features = [f for f in features if f not in date_features]
 
if other_features:
    st.markdown("#### 📊 Other Features")
    cols = st.columns(2)
    for i, feature in enumerate(other_features):
        with cols[i % 2]:
            input_data[feature] = st.number_input(
                label=feature,
                value=0.0,
                format="%.4f",
                key=feature
            )
 
# ===================== PREDICT =====================
st.markdown("---")
if st.button("🔮 Predict Sales", type="primary", use_container_width=True):
    try:
        # Build DataFrame in the exact feature order the model expects
        input_df = pd.DataFrame([{f: input_data[f] for f in features}])
 
        st.markdown("#### 🧾 Input Summary")
        st.dataframe(input_df, use_container_width=True)
 
        prediction = model.predict(input_df)[0]
 
        st.markdown("---")
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius: 16px;
                padding: 36px;
                text-align: center;
                border: 1px solid #0f3460;
                box-shadow: 0 4px 30px rgba(0,0,0,0.3);
            ">
                <p style="color:#a0aec0; font-size:16px; margin-bottom:8px;">Predicted Sales</p>
                <h1 style="color:#63b3ed; font-size:52px; margin:0;">💰 {prediction:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
 
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.exception(e)
 
# ===================== FOOTER =====================
st.markdown("---")
st.caption("Model: Random Forest  |  Files required: model1.joblib + features1.joblib")