import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Hybrid ML Machine Health Dashboard",
    layout="wide",
    page_icon="🛠️"
)
st.title("🛠️ Hybrid Machine Health Monitoring Dashboard")

# -----------------------------
# Paths
# -----------------------------
BASE = Path(".")
AE_PATH = BASE / "Advanced_Hybrid_ML_Project" / "models" / "autoencoder_model"
LSTM_PATH = BASE / "Advanced_Hybrid_ML_Project" / "models" / "lstm_rul_model"
FUSION_MODEL_PATH = BASE / "Advanced_Hybrid_ML_Project" / "models" / "fusion_model_joblib.pkl"
FUSION_SCALER_PATH = BASE / "Advanced_Hybrid_ML_Project" / "models" / "fusion_feature_scaler_joblib.pkl"
GLOBAL_MINMAX_PATH = BASE / "Advanced_Hybrid_ML_Project" / "models" / "global_minmax_joblib.pkl"
DASHBOARD_CSV = BASE / "Advanced_Hybrid_ML_Project" / "data" / "raw" / "dashboard_dataset.csv"

WINDOW = 100  # window size used in training

# -----------------------------
# Display helper
# -----------------------------
def colored_badge(text, color):
    return f"""
    <span style="background-color:{color};
                 padding:6px 12px;
                 border-radius:10px;
                 color:white;
                 font-weight:700;">
        {text}
    </span>
    """

def plot_series(y, title, color="blue"):
    fig = plt.figure(figsize=(12, 3))
    plt.plot(y, color=color)
    plt.title(title)
    plt.grid(True)
    st.pyplot(fig)

def st_error_and_stop(msg):
    st.error(msg)
    st.stop()

# -----------------------------
# Load models
# -----------------------------
st.sidebar.info("Loading models...")

try:
    autoencoder = tf.keras.models.load_model(str(AE_PATH))
    lstm_rul = tf.keras.models.load_model(str(LSTM_PATH))
except Exception as e:
    st_error_and_stop(f"Failed to load TensorFlow models: {e}")

try:
    fusion_model = joblib.load(str(FUSION_MODEL_PATH))
    fusion_scaler = joblib.load(str(FUSION_SCALER_PATH))
except Exception as e:
    st_error_and_stop(f"Failed to load fusion model artifacts: {e}")

# Load global min/max
try:
    gm = joblib.load(str(GLOBAL_MINMAX_PATH))
    if isinstance(gm, (tuple, list)):
        Xmin, Xmax = np.array(gm[0]), np.array(gm[1])
    else:
        Xmin = np.array(gm["min"] if "min" in gm else gm["Xmin"])
        Xmax = np.array(gm["max"] if "max" in gm else gm["Xmax"])
except Exception as e:
    st_error_and_stop(f"Failed to load global min/max: {e}")

st.sidebar.success("Models loaded successfully ✔")

# load dashboard dataset
dashboard_df = pd.read_csv(DASHBOARD_CSV) if DASHBOARD_CSV.exists() else pd.DataFrame()

# -----------------------------
# Preprocess functions
# -----------------------------
def scale_raw_sensors(data):
    return (data - Xmin) / (Xmax - Xmin + 1e-9)

def create_windows(data, window=WINDOW):
    if len(data) < window:
        pad = np.repeat(data[-1:].reshape(1, -1), window - len(data), axis=0)
        data = np.vstack([data, pad])
    return np.array([data[i:i + window] for i in range(len(data) - window + 1)])

# -----------------------------
# UI Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📊 Dashboard", "📁 Upload & Predict"])


# ----------------------------------------------------------
# TAB 2 — UPLOAD & PREDICT (MOST IMPORTANT PART)
# ----------------------------------------------------------
with tab2:
    st.subheader("📁 Upload Sensor CSV for Real-Time Prediction")

    uploaded = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(df.head())

        required = ["vibration","temperature","pressure","torque","current","rpm"]
        if not set(required).issubset(df.columns):
            st_error_and_stop("CSV missing required columns")

        raw = df[required].values.astype(float)
        scaled = scale_raw_sensors(raw)
        windows = create_windows(scaled)

        # AE anomaly prediction
        X_rec = autoencoder.predict(windows, verbose=0)
        anomaly_scores = np.mean((windows - X_rec)**2, axis=(1, 2))
        latest_anom = float(anomaly_scores[-1])

        # LSTM RUL prediction
        lstm_raw = lstm_rul.predict(windows, verbose=0).flatten()
        if np.max(lstm_raw) <= 1.01:
            rul_vals = lstm_raw * max(len(dashboard_df), 5000)
        else:
            rul_vals = lstm_raw
        rul_vals = np.maximum(rul_vals, 0)
        latest_rul = float(rul_vals[-1])

        # Fusion prediction
        fusion_features = np.array([[latest_anom, latest_rul]])
        fusion_scaled = fusion_scaler.transform(fusion_features)
        failure_prob = float(fusion_model.predict_proba(fusion_scaled)[0][1])

        # --------------------------
        # 🔥 HEALTH INDEX FIX (FINAL)
        # --------------------------

        # Soft anomaly scaling (makes AE less aggressive)
        a_mean = float(np.mean(anomaly_scores))
        a_std = float(np.std(anomaly_scores) + 1e-9)
        a_z = (latest_anom - a_mean) / a_std
        an_norm = np.clip((a_z + 4) / 10.0, 0, 1)

        # RUL dominant weighting (more stable)
        rul_norm = np.clip(latest_rul / max(len(dashboard_df), 5000), 0, 1)

        health_index = float(np.clip(0.25*(1-an_norm) + 0.75*rul_norm, 0, 1))

        # Machine status
        if health_index > 0.7:
            status = "HEALTHY"; color = "green"
        elif health_index > 0.4:
            status = "WARNING"; color = "orange"
        else:
            status = "CRITICAL FAILURE"; color = "red"

        # RUL interpretation
        if latest_rul > 3000:
            rul_state = "Long life remaining"; rul_color = "green"
        elif latest_rul > 1000:
            rul_state = "Mid-life (Degrading)"; rul_color = "orange"
        else:
            rul_state = "End-of-Life Soon"; rul_color = "red"

        # --------------------------
        # UI Display
        # --------------------------
        st.subheader("🔍 Prediction Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Health Index", f"{health_index:.2f}")
        c1.markdown(colored_badge(status, color), unsafe_allow_html=True)

        c2.metric("Failure Probability", f"{failure_prob:.2f}")
        c2.markdown(colored_badge(
            "Safe (<0.3)" if failure_prob < 0.3 else 
            "Risky (0.3–0.7)" if failure_prob < 0.7 else 
            "High Risk (>0.7)",
            "green" if failure_prob < 0.3 else "orange" if failure_prob < 0.7 else "red"
        ), unsafe_allow_html=True)

        c3.metric("Predicted RUL", f"{latest_rul:.0f} steps")
        c3.markdown(colored_badge(rul_state, rul_color), unsafe_allow_html=True)


        st.markdown("---")

        # --------------------------
        # NEW: Overall machine state
        # --------------------------
        st.markdown(f"### 🏭 Machine Status: {colored_badge(status, color)}", unsafe_allow_html=True)

        st.markdown("---")

        plot_series(anomaly_scores, "Anomaly Score", "orange")
        plot_series(rul_vals, "RUL Prediction", "purple")
