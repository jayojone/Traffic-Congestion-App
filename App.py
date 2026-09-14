import streamlit as st
import pandas as pd
import joblib
import json
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Dar es Salaam Traffic Predictor",
    page_icon="🚦",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM STYLING ----------
st.markdown("""
    <style>
    .main { background-color: #f7f9fc; }
    .welcome-box {
        background: linear-gradient(135deg, #1a2b4c, #2e7bf6);
        padding: 1.6rem 1.8rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .welcome-title { font-size: 1.7rem; font-weight: 800; margin-bottom: 0.3rem; }
    .welcome-sub { font-size: 0.98rem; color: #dbe6ff; line-height: 1.5; }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a2b4c;
        margin-top: 1.5rem;
        margin-bottom: 0.3rem;
        border-left: 4px solid #2e7bf6;
        padding-left: 8px;
    }
    .stButton>button {
        background-color: #2e7bf6;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #1a5fd0; color: white; }
    .result-card {
        padding: 1.2rem; border-radius: 12px; text-align: center;
        margin-top: 1rem; font-size: 1.4rem; font-weight: 800;
    }
    .low   { background-color: #d4f8e8; color: #1a7a4c; }
    .moderate { background-color: #fff4cc; color: #8a6d00; }
    .high  { background-color: #ffe0cc; color: #b04a00; }
    .veryhigh { background-color: #ffd6d6; color: #a30000; }
    .caption-img { font-size: 0.8rem; color: #6b7385; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# ---------- SIMPLE STATS STORAGE (views & likes) ----------
STATS_FILE = "stats.json"

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    return {"views": 0, "likes": 0}

def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

stats = load_stats()

# Count a view only once per browser session
if "counted" not in st.session_state:
    stats["views"] += 1
    save_stats(stats)
    st.session_state["counted"] = True

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("congestion_model.pkl")
    encoders = joblib.load("encoders.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, encoders, model_columns

model, encoders, model_columns = load_artifacts()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 🚦 About this App")
    st.write(
        "This tool predicts the **traffic congestion level** on major roads "
        "in Dar es Salaam using a Machine Learning model (Random Forest) "
        "trained on traffic patterns across time, location, and weather."
    )
    st.markdown("---")
    st.markdown("### 📊 How it works")
    st.write(
        "1. Choose the road, day, and time.\n"
        "2. Enter the current traffic conditions.\n"
        "3. Click **Predict** to see the expected congestion level."
    )
    st.markdown("---")
    st.markdown("### 🎓 Project Info")
    st.caption(
        "Data Science practical attachment project — "
        "Predicting and Analyzing Traffic Congestion in Dar es Salaam using "
        "Data Science and Machine Learning."
    )

    # ---------- ADMIN LOGIN (hidden section) ----------
    st.markdown("---")
    with st.expander("🔐 Admin Login"):
        admin_user = st.text_input("Username", key="admin_user")
        admin_pass = st.text_input("Password", type="password", key="admin_pass")
        if st.button("Login", key="admin_login_btn"):
            # ⚠️ Change these credentials before submitting/sharing your project publicly
            if admin_user == "japhet" and admin_pass == "admin2026":
                st.session_state["is_admin"] = True
            else:
                st.session_state["is_admin"] = False
                st.error("Incorrect username or password.")

        if st.session_state.get("is_admin"):
            st.success("Logged in as Admin")
            st.metric("Total Views", stats["views"])
            st.metric("Total Appreciations", stats["likes"])

# ---------- WELCOME BANNER ----------
st.markdown("""
    <div class="welcome-box">
        <div class="welcome-title">👋 Welcome to the Dar es Salaam Traffic Congestion Predictor</div>
        <div class="welcome-sub">
            This platform applies Data Science and Machine Learning to help commuters, planners,
            and researchers anticipate traffic congestion across major roads in Dar es Salaam.
            Simply provide a few details about the road, time, and current conditions, and the
            system will instantly forecast the expected congestion level — empowering smarter,
            data-driven travel decisions.
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------- ILLUSTRATIVE IMAGES SECTION ----------
st.markdown('<p class="section-header">🖼️ Understanding Congestion Hotspots</p>', unsafe_allow_html=True)
st.caption(
    "Representative images of busy roads included in this study. "
    "(Place your own images inside an `images/` folder next to App.py — see note below.)"
)

image_info = [
    ("images/mwenge.jpg", "Mwenge — a major northern junction"),
    ("images/ubungo.jpg", "Ubungo — a key interchange linking multiple highways"),
    ("images/kariakoo.jpg", "Kariakoo — dense commercial-area traffic"),
]

cols = st.columns(3)
for col, (path, caption) in zip(cols, image_info):
    with col:
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info("Image not found — add it to the `images/` folder.")
        st.markdown(f'<p class="caption-img">{caption}</p>', unsafe_allow_html=True)

# ---------- SECTION 1: ROAD & TIME ----------
st.markdown('<p class="section-header">📍 Road & Time Details</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("Location", encoders["Location"].classes_,
                             help="Choose the road or junction you want to check.")
    day = st.selectbox("Day of Week", encoders["Day_of_Week"].classes_,
                        help="Traffic behaves differently on weekdays vs weekends.")
with col2:
    hour = st.selectbox("Hour of Day", [6, 7, 8, 12, 17, 18],
                         help="6-8 and 17-18 are typical morning/evening rush hours.")
    road_condition = st.selectbox("Road Condition", encoders["Road_Condition"].classes_)

# ---------- SECTION 2: CONDITIONS ----------
st.markdown('<p class="section-header">🌦️ Conditions</p>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    weather = st.selectbox("Weather", encoders["Weather"].classes_)
with col4:
    accident = st.selectbox("Accident Occurring?", encoders["Accident"].classes_)

# ---------- SECTION 3: TRAFFIC MEASUREMENTS ----------
st.markdown('<p class="section-header">📈 Traffic Measurements</p>', unsafe_allow_html=True)
st.caption("If you don't have exact numbers, use the default estimates provided.")
col5, col6, col7 = st.columns(3)
with col5:
    traffic_volume = st.number_input("Traffic Volume", min_value=0, max_value=400, value=150,
                                      help="Estimated number of vehicles passing per interval.")
with col6:
    avg_speed = st.number_input("Avg Speed (km/h)", min_value=0.0, max_value=60.0, value=20.0)
with col7:
    travel_time = st.number_input("Travel Time /10km (min)", min_value=0.0, max_value=100.0, value=25.0)

is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
peak_hour = "Yes" if hour in [6, 7, 8, 17, 18] else "No"

st.markdown("---")
st.caption(f"Detected automatically → Weekend: **{'Yes' if is_weekend else 'No'}** • Peak Hour: **{peak_hour}**")

# ---------- PREDICTION ----------
if st.button("🔍 Predict Congestion Level"):
    input_dict = {
        "Location": encoders["Location"].transform([location])[0],
        "Weekend": is_weekend,
        "Traffic_Volume": traffic_volume,
        "Average_Speed_kmh": avg_speed,
        "Travel_Time_min_10km": travel_time,
        "Weather": encoders["Weather"].transform([weather])[0],
        "Accident": encoders["Accident"].transform([accident])[0],
        "Road_Condition": encoders["Road_Condition"].transform([road_condition])[0],
        "Hour": hour,
        "Peak_Hour": encoders["Peak_Hour"].transform([peak_hour])[0],
        "Day_of_Week": encoders["Day_of_Week"].transform([day])[0],
    }

    input_df = pd.DataFrame([input_dict])[model_columns]
    prediction = model.predict(input_df)[0]
    predicted_label = encoders["Congestion_Level"].inverse_transform([prediction])[0]

    style_map = {
        "Low": ("low", "🟢"),
        "Moderate": ("moderate", "🟡"),
        "High": ("high", "🟠"),
        "Very High": ("veryhigh", "🔴"),
    }
    css_class, emoji = style_map.get(predicted_label, ("moderate", "🟡"))

    st.markdown(
        f'<div class="result-card {css_class}">{emoji} Predicted Congestion Level: {predicted_label}</div>',
        unsafe_allow_html=True
    )

    advice = {
        "Low": "Traffic is flowing freely — a good time to travel on this route.",
        "Moderate": "Some slowdowns expected — allow a little extra travel time.",
        "High": "Significant congestion likely — consider an alternative route or time.",
        "Very High": "Severe congestion expected — strongly consider delaying travel or rerouting.",
    }
    st.info(advice.get(predicted_label, ""))

# ---------- APPRECIATION BUTTON (public) ----------
st.markdown("---")
colA, colB = st.columns([3, 1])
with colA:
    st.caption("Found this project useful? Let the developer know 👇")
with colB:
    if st.button("👍 Appreciate"):
        stats["likes"] += 1
        save_stats(stats)
        st.success("Thank you!")

st.caption("Built as part of a Data Science practical attachment project • Model: Random Forest Classifier")
