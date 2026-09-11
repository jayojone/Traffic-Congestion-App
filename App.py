import streamlit as st
import pandas as pd
import joblib

# Load the saved model, encoders, and column order
model = joblib.load("congestion_model.pkl")
encoders = joblib.load("encoders.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Dar es Salaam Traffic Congestion Predictor")
st.write("Enter the conditions below to predict the expected traffic congestion level.")

# --- User inputs ---
location = st.selectbox("Location", encoders["Location"].classes_)
day = st.selectbox("Day of Week", encoders["Day_of_Week"].classes_)
hour = st.selectbox("Hour of Day", [6, 7, 8, 12, 17, 18])
weather = st.selectbox("Weather", encoders["Weather"].classes_)
accident = st.selectbox("Accident Occurring?", encoders["Accident"].classes_)
road_condition = st.selectbox("Road Condition", encoders["Road_Condition"].classes_)
traffic_volume = st.number_input("Traffic Volume", min_value=0, max_value=400, value=150)
avg_speed = st.number_input("Average Speed (km/h)", min_value=0.0, max_value=60.0, value=20.0)
travel_time = st.number_input("Travel Time for 10km (minutes)", min_value=0.0, max_value=100.0, value=25.0)

# --- Auto-derived fields ---
is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
peak_hour = "Yes" if hour in [6, 7, 8, 17, 18] else "No"

if st.button("Predict Congestion Level"):
    # Build input row matching training column order
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

    st.subheader(f"Predicted Congestion Level: **{predicted_label}**")