import streamlit as st
import pickle
import numpy as np
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .fraud-header {
        color: #1E3D59;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# Load model and scaler
@st.cache_resource
def load_models():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return scaler, model


scaler, model = load_models()

# Header
st.markdown("<h1 class='fraud-header'>🔒 Credit Card Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Transaction Details")
    with st.container():
        cc_num = st.text_input("Credit Card Number", placeholder="Enter 16-digit card number")
        amt = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")

        # Transaction DateTime
        st.subheader("🕒 Transaction Time")
        col_date, col_time = st.columns(2)
        with col_date:
            date = st.date_input("Select Date")
        with col_time:
            hour = st.slider("Hour (24-hr format)", 0, 23, 12)

with col2:
    st.subheader("👤 Customer Information")
    gender = st.selectbox("Gender", ["Select Gender", "Male", "Female"])
    gender_map = {"Male": 1, "Female": 0}
    gender_val = gender_map.get(gender, 0)

    st.subheader("📍 Location Details")
    lat = st.number_input("Customer Latitude", format="%.6f")
    long = st.number_input("Customer Longitude", format="%.6f")
    city_pop = st.number_input("City Population", min_value=0)

# Merchant Information in Expander
with st.expander("🏪 Merchant Information", expanded=False):
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        merchant = st.number_input("Merchant ID", min_value=0)
        category = st.selectbox("Category",
                                ["Retail", "Grocery", "Entertainment", "Travel", "Restaurant", "Other"])
        category_map = {"Retail": 0, "Grocery": 1, "Entertainment": 2,
                        "Travel": 3, "Restaurant": 4, "Other": 5}
        category_val = category_map.get(category, 0)
    with col_m2:
        merch_lat = st.number_input("Merchant Latitude", format="%.6f")
        merch_long = st.number_input("Merchant Longitude", format="%.6f")

# Additional Details (Hidden in expander)
with st.expander("➕ Additional Details", expanded=False):
    job = st.number_input("Job ID", min_value=0)
    unix_time = int(datetime.combine(date, datetime.min.time()).timestamp()) + (hour * 3600)
    day = date.day
    month = date.month

# Create analysis button with custom styling
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Transaction", use_container_width=True)

if analyze_button and cc_num and amt:  # Basic validation
    # Progress bar for visual feedback
    progress_text = "Analyzing transaction..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        my_bar.progress(percent_complete + 1, text=progress_text)

    # Prepare input data
    input_data = np.array([[
        float(cc_num), merchant, category_val, amt, gender_val,
        lat, long, city_pop, job, unix_time, merch_lat, merch_long,
        hour, day, month
    ]])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    # Show result with custom styling
    st.markdown("### 📝 Analysis Result")
    if prediction[0] == 1:
        st.error("""
        ⚠️ **FRAUDULENT TRANSACTION DETECTED**

        This transaction has been flagged as potentially fraudulent.
        Recommended actions:
        - Verify the transaction with the cardholder
        - Check for unusual patterns
        - Consider blocking the card temporarily
        """)
    else:
        st.success("""
        ✅ **LEGITIMATE TRANSACTION**

        This transaction appears to be normal and legitimate.
        - Risk level: Low
        - Transaction can be processed
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666666; padding: 1rem;'>
        Powered by Advanced Machine Learning | © 2025 Fraud Detection System
    </div>
""", unsafe_allow_html=True)