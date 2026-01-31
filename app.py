import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ===== Custom CSS =====
st.markdown("""
<style>
/* Page background */
.stApp {
    background-color: #f4f6f9;
}

/* Main card */
.main-card {
    background-color: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

/* Title */
.title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: #1f2d3d;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #5a6c7d;
    margin-bottom: 30px;
}

/* Button */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: 600;
}

/* Result box */
.result-box {
    background-color: #f0f7ff;
    border-left: 6px solid #2563eb;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ===== UI =====

st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title"> House Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Professional Machine Learning Web Application</div>', unsafe_allow_html=True)

# Inputs
col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input(
    "Median Income (×10,000$)",
    min_value=0.0,
    value=5.0,
    help="This value is multiplied by 10,000$. Example: 5 = 50,000$ per year")

    st.caption(f" This means approximately: {MedInc * 10000:,.0f} $ per year")
    HouseAge = st.number_input("House Age", min_value=0.0, value=20.0)
    AveRooms = st.number_input("Average Rooms", min_value=0.0, value=5.0)
    AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0)

with col2:
    Population = st.number_input("Population", min_value=1.0, value=1000.0)
    AveOccup = st.number_input("Average Occupancy", min_value=0.0, value=3.0)
    Latitude = st.number_input("Latitude", value=34.0)
    Longitude = st.number_input("Longitude", value=-118.0)

st.markdown("<br>", unsafe_allow_html=True)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)

    st.markdown(f"""
    <div class="result-box">
        <h3>Estimated House Price</h3>
        <h1 style="color:#2563eb;">${prediction[0]*100000:,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Built by Teslem El Maazouz | Machine Learning Project")
