import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
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
# MODEL PATHS
# ---------------------------------------
AE_PATH = "Advanced_Hybrid_ML_Project/models/autoencoder_model"
LSTM_PATH = "Advanced_Hybrid_ML_Project/models/lstm_rul_model"
FUSION_MODEL_PATH = "Advanced_Hybrid_ML_Project/models/fusion_model_joblib.pkl"
FUSION_SCALER_PATH = "Advanced_Hybrid_ML_Project/models/fusion_feature_scaler_joblib.pkl"
GLOBAL_MINMAX_PATH = "Advanced_Hybrid_ML_Project/models/global_minmax_joblib.pkl"

# ---------------------------------------
# LOAD MODELS
# ---------------------------------------
autoencoder = tf.keras.models.load_model(AE_PATH)
lstm_rul = tf.keras.models.load_model(LSTM_PATH)

fusion_model = joblib.load(FUSION_MODEL_PATH)
fusion_scaler = joblib.load(FUSION_SCALER_PATH)
Xmin, Xmax = joblib.load(GLOBAL_MINMAX_PATH)

# Load dashboard dataset
dashboard_df = pd.read_csv("Advanced_Hybrid_ML_Project/data/raw/dashboard_dataset.csv")

# ---------------------------------------
# FUNCTIONS
# ---------------------------------------
def plot_series(y, title, color="blue"):
    fig = plt.figure(figsize=(12, 4))
    plt.plot(y, color=color)
    plt.title(title)
    plt.grid(True)
    st.pyplot(fig)

def preprocess_data(df):
    required = ["vibration", "temperature", "pressure", "torque", "current", "rpm"]
    X = df[required].values.astype(float)

    # Scale using saved min/max
    X_scaled = (X - Xmin) / (Xmax - Xmin + 1e-9)

    # Windowing
    window = 100
    if len(X_scaled) < window:
        pad = np.repeat(X_scaled[-1:], window - len(X_scaled), axis=0)
        X_scaled = np.vstack([X_scaled, pad])

    Xw = []
    for i in range(len(X_scaled) - window + 1):
        Xw.append(X_scaled[i:i + window])

    return np.array(Xw)

# ---------------------------------------
# MAIN TABS
# ---------------------------------------
tab1, tab2 = st.tabs(["📊 Dashboard", "📁 Upload & Predict"])

# ============================================================
# TAB 1 — DASHBOARD VIEW
# ============================================================
with tab1:

    st.subheader("📊 Machine Health Overview")

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
            st.error("❌ CSV missing required columns")
        else:
            Xw = preprocess_data(user_df)

            # --- Hybrid Predictions ---
            # AE anomaly score
            X_rec = autoencoder.predict(Xw, verbose=0)
            anomaly_scores = np.mean((Xw - X_rec)**2, axis=(1, 2))
            anomaly_score = float(anomaly_scores[-1])

            # LSTM RUL
            rul_values = lstm_rul.predict(Xw, verbose=0).flatten()
            rul_pred = float(rul_values[-1])

            # Fusion failure probability
            fusion_input = fusion_scaler.transform([[anomaly_score, rul_pred]])
            failure_probability = float(fusion_model.predict_proba(fusion_input)[0][1])

            # Health index
            health_index = max(0.0, 1 - anomaly_score * 50)
            health_index = min(1.0, health_index)

            st.subheader("🔍 Prediction Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Health Index", f"{health_index:.2f}")
            col2.metric("Failure Probability", f"{failure_probability:.2f}")
            col3.metric("Predicted RUL", f"{rul_pred:.0f} steps")

            st.markdown("---")

            # Plots
            plot_series([health_index], "Health Index (Uploaded Data)", color="green")
            plot_series([failure_probability], "Failure Probability (Uploaded Data)", color="red")
            plot_series(anomaly_scores, "Anomaly Score (Uploaded Data)", color="orange")
            plot_series(rul_values, "RUL Prediction (Uploaded Data)", color="purple")
