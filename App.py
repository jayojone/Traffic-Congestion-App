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
    .loc-name { font-size: 0.95rem; font-weight: 700; color: #1a2b4c; text-align: center; margin-top: 0.4rem; }
    .loc-badge { font-size: 0.75rem; text-align: center; font-weight: 700; margin-bottom: 0.6rem; }
    .meter-track { background-color: #e6e9f0; border-radius: 6px; height: 8px; width: 100%; overflow: hidden; }
    .meter-fill { height: 8px; border-radius: 6px; }
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

# ---------- ORIGINAL SVG SCENES (self-drawn, no external images) ----------

def car(x, y, w, h, body_color, scale=1.0):
    return f"""
    <g transform="translate({x},{y}) scale({scale})">
        <rect x="0" y="0" width="{w}" height="{h}" rx="2.5" fill="{body_color}"/>
        <rect x="{w*0.15}" y="-{h*0.35}" width="{w*0.7}" height="{h*0.4}" rx="2" fill="{body_color}" opacity="0.85"/>
        <circle cx="{w*0.22}" cy="{h}" r="{h*0.32}" fill="#22283a"/>
        <circle cx="{w*0.78}" cy="{h}" r="{h*0.32}" fill="#22283a"/>
    </g>
    """

def ubungo_illustration():
    cars_top = "".join(
        car(24 + i * 26, 58, 20, 10, c) for i, c in enumerate(
            ["#e35b5b", "#5b7fd6", "#f2a33c", "#e35b5b", "#5b7fd6", "#f2a33c", "#e35b5b", "#5b7fd6", "#f2a33c"]
        )
    )
    cars_bottom = "".join(
        car(20 + i * 27, 128, 20, 10, c) for i, c in enumerate(
            ["#f2a33c", "#e35b5b", "#5b7fd6", "#f2a33c", "#e35b5b", "#5b7fd6", "#f2a33c", "#e35b5b"]
        )
    )
    return f"""
    <svg viewBox="0 0 300 170" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;border-radius:10px;">
        <rect x="0" y="0" width="300" height="170" fill="#dfe8ff"/>
        <rect x="14" y="12" width="28" height="55" fill="#ffffff" opacity="0.55"/>
        <rect x="250" y="18" width="30" height="50" fill="#ffffff" opacity="0.5"/>
        <path d="M0 95 Q150 55 300 95 L300 115 Q150 75 0 115 Z" fill="#8b93a8"/>
        <rect x="30" y="115" width="8" height="45" fill="#6b7285"/>
        <rect x="140" y="105" width="8" height="55" fill="#6b7285"/>
        <rect x="250" y="115" width="8" height="45" fill="#6b7285"/>
        <rect x="0" y="140" width="300" height="30" fill="#9aa1b5"/>
        {cars_top}
        {cars_bottom}
    </svg>
    """

def kariakoo_illustration():
    cars = "".join(
        car(16 + i * 24, 112, 18, 10, c) for i, c in enumerate(
            ["#5b7fd6", "#f2a33c", "#e35b5b", "#5b7fd6", "#f2a33c", "#e35b5b", "#5b7fd6", "#f2a33c", "#e35b5b", "#5b7fd6"]
        )
    )
    return f"""
    <svg viewBox="0 0 300 170" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;border-radius:10px;">
        <rect x="0" y="0" width="300" height="170" fill="#ffe9d6"/>
        <rect x="0" y="20" width="60" height="70" fill="#d68a5b"/>
        <rect x="65" y="14" width="55" height="76" fill="#e0996a"/>
        <rect x="185" y="16" width="50" height="74" fill="#d68a5b"/>
        <rect x="240" y="20" width="60" height="70" fill="#e0996a"/>
        <rect x="8" y="30" width="12" height="12" fill="#fff4e0"/>
        <rect x="78" y="26" width="12" height="12" fill="#fff4e0"/>
        <rect x="196" y="28" width="12" height="12" fill="#fff4e0"/>
        <rect x="256" y="30" width="12" height="12" fill="#fff4e0"/>
        <rect x="0" y="95" width="300" height="75" fill="#8b8f9c"/>
        <line x1="0" y1="132" x2="300" y2="132" stroke="#ffffff" stroke-width="2" stroke-dasharray="8,6"/>
        {cars}
    </svg>
    """

def mwenge_illustration():
    cars = "".join(
        car(30 + i * 62, 118, 20, 11, c) for i, c in enumerate(
            ["#5b9fd6", "#d6c05b", "#5bd68f"]
        )
    )
    return f"""
    <svg viewBox="0 0 300 170" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;border-radius:10px;">
        <rect x="0" y="0" width="300" height="170" fill="#fff6d6"/>
        <rect x="20" y="18" width="30" height="60" fill="#ffffff" opacity="0.55"/>
        <rect x="245" y="14" width="34" height="64" fill="#ffffff" opacity="0.5"/>
        <rect x="0" y="95" width="300" height="75" fill="#a3a8ba"/>
        <line x1="0" y1="132" x2="300" y2="132" stroke="#ffffff" stroke-width="2" stroke-dasharray="10,10"/>
        <circle cx="150" cy="132" r="26" fill="#8f95a8"/>
        <circle cx="150" cy="132" r="12" fill="#c7d68f"/>
        {cars}
    </svg>
    """

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

# ---------- ILLUSTRATIVE SECTION (original artwork, no external images) ----------
st.markdown('<p class="section-header">🖼️ Congestion Hotspots at a Glance</p>', unsafe_allow_html=True)
st.caption("Original illustrations depicting typical scenes and congestion severity found in this study's analysis.")

hotspots = [
    ("Ubungo Flyover", ubungo_illustration, "Very High", "#a30000", 95),
    ("Kariakoo", kariakoo_illustration, "High", "#b04a00", 75),
    ("Mwenge", mwenge_illustration, "Moderate", "#8a6d00", 45),
]

cols = st.columns(3)
for col, (name, illustration_fn, level, color, pct) in zip(cols, hotspots):
    with col:
        st.markdown(illustration_fn(), unsafe_allow_html=True)
        st.markdown(f'<p class="loc-name">{name}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="loc-badge" style="color:{color};">Typical congestion: {level}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="meter-track"><div class="meter-fill" style="width:{pct}%;background-color:{color};"></div></div>',
            unsafe_allow_html=True
        )

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
