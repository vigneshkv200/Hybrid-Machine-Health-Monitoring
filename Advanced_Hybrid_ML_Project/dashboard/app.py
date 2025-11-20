import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(
    page_title="Hybrid ML Machine Health Dashboard",
    layout="wide",
    page_icon="🛠️",
)

st.title("🛠️ Hybrid Machine Health Monitoring Dashboard")

# ---------------------------------------
# LOAD MODELS
# ---------------------------------------
autoencoder = tf.keras.models.load_model("Advanced_Hybrid_ML_Project/models/autoencoder_model")
lstm_rul = tf.keras.models.load_model("Advanced_Hybrid_ML_Project/models/lstm_rul_model")

with open("Advanced_Hybrid_ML_Project/models/fusion_model.pkl", "rb") as f:
    fusion_model = pickle.load(f)

with open("Advanced_Hybrid_ML_Project/models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load dashboard dataset
dashboard_df = pd.read_csv("Advanced_Hybrid_ML_Project/data/raw/dashboard_dataset.csv")

# ---------------------------------------
# FUNCTIONS
# ---------------------------------------
def plot_series(y, title, color="blue"):
    fig = plt.figure(figsize=(12,4))
    plt.plot(y, color=color)
    plt.title(title)
    plt.grid(True)
    st.pyplot(fig)

def create_windows(data, window=100):
    X = []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
    return np.array(X)

# ---------------------------------------
# MAIN TABS
# ---------------------------------------
tab1, tab2 = st.tabs(["📊 Dashboard", "📁 Upload & Predict"])

# ============================================================
# TAB 1 — DASHBOARD VIEW
# ============================================================
with tab1:

    st.subheader("📊 Machine Health Overview")

    # Health Status Card
    current_health = dashboard_df["health_index"].iloc[-1]

    if current_health > 0.7:
        color = "green"
        status = "HEALTHY"
    elif current_health > 0.4:
        color = "orange"
        status = "WARNING"
    else:
        color = "red"
        status = "CRITICAL FAILURE"

    st.markdown(
        f"""
        <div style="padding:15px; border-radius:10px; background-color:#f0f0f0; text-align:center;">
            <h2 style='color:{color};'>Machine Status: {status}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Health Index", f"{current_health:.2f}")
    col2.metric("Failure Probability", f"{dashboard_df['failure_probability'].iloc[-1]:.2f}")
    col3.metric("Predicted RUL", f"{dashboard_df['rul_prediction'].iloc[-1]:.0f} steps")

    st.markdown("---")

    # Visualization Options
    option = st.radio(
        "Select Visualization:",
        ["Health Index", "Failure Probability", "RUL Prediction", "Anomaly Score", "Raw Sensors"],
        horizontal=True
    )

    if option == "Health Index":
        plot_series(dashboard_df["health_index"], "Health Index Over Time", color="green")

    elif option == "Failure Probability":
        plot_series(dashboard_df["failure_probability"], "Failure Probability Over Time", color="red")

    elif option == "RUL Prediction":
        plot_series(dashboard_df["rul_prediction"], "RUL Prediction Over Time", color="purple")

    elif option == "Anomaly Score":
        plot_series(dashboard_df["anomaly_score"], "Anomaly Score Timeline", color="orange")

    elif option == "Raw Sensors":
        sensor = st.selectbox(
            "Choose Sensor:",
            ["vibration", "temperature", "pressure", "torque", "current", "rpm"]
        )
        plot_series(dashboard_df[sensor], f"{sensor.capitalize()} Over Time", color="blue")

# ============================================================
# TAB 2 — UPLOAD & PREDICT
# ============================================================
with tab2:

    st.subheader("📁 Upload Sensor CSV for Real-Time Prediction")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write(user_df.head())

        required_cols = ["vibration", "temperature", "pressure", "torque", "current", "rpm"]

        if not set(required_cols).issubset(user_df.columns):
            st.error("❌ CSV missing required sensor columns: vibration, temperature, pressure, torque, current, rpm")
        else:
            sensor_data = user_df[required_cols].values
            sensor_scaled = scaler.transform(sensor_data)
            X_input = create_windows(sensor_scaled)

            # AE Predict
            X_rec = autoencoder.predict(X_input)
            anomaly_scores = np.mean((X_input - X_rec)**2, axis=(1,2))

            # LSTM Predict
            rul_scaled_pred = lstm_rul.predict(X_input).flatten()
            rul_pred = rul_scaled_pred * 9900

            # Fusion Predict
            fusion_features = np.column_stack([anomaly_scores, rul_pred])
            failure_prob_user = fusion_model.predict_proba(fusion_features)[:, 1]

            # Health Index
            anomaly_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
            health_index_user = 0.5 * (1 - anomaly_norm) + 0.5 * rul_scaled_pred

            st.subheader("🔍 Prediction Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Health Index", f"{health_index_user[-1]:.2f}")
            col2.metric("Failure Probability", f"{failure_prob_user[-1]:.2f}")
            col3.metric("Predicted RUL", f"{rul_pred[-1]:.0f} steps")

            st.markdown("---")

            # Plots
            plot_series(health_index_user, "Health Index (Uploaded Data)", color="green")
            plot_series(failure_prob_user, "Failure Probability (Uploaded Data)", color="red")
            plot_series(anomaly_scores, "Anomaly Score (Uploaded Data)", color="orange")
            plot_series(rul_pred, "RUL Prediction (Uploaded Data)", color="purple")


