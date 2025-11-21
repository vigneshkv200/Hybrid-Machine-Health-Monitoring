import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hybrid ML Machine Health Dashboard",
                   layout="wide",
                   page_icon="🛠️")

# ---------------------------------------------------------
# LOAD MODELS (NEW UPDATED FILES)
# ---------------------------------------------------------

AUTOENCODER_PATH = "Advanced_Hybrid_ML_Project/models/autoencoder_model"
LSTM_MODEL_PATH = "Advanced_Hybrid_ML_Project/models/lstm_rul_model"
FUSION_MODEL_PATH = "Advanced_Hybrid_ML_Project/models/fusion_model_joblib.pkl"
FUSION_SCALER_PATH = "Advanced_Hybrid_ML_Project/models/fusion_feature_scaler_joblib.pkl"
GLOBAL_MINMAX_PATH = "Advanced_Hybrid_ML_Project/models/global_minmax_joblib.pkl"

st.sidebar.success("Models Loading...")

autoencoder = tf.keras.models.load_model(AUTOENCODER_PATH)
lstm_rul = tf.keras.models.load_model(LSTM_MODEL_PATH)
fusion_model = joblib.load(FUSION_MODEL_PATH)
fusion_scaler = joblib.load(FUSION_SCALER_PATH)


global_minmax = joblib.load(GLOBAL_MINMAX_PATH)

# Support both possible saved formats:
# - older code saved (Xmin, Xmax) as a tuple
# - newer code may save {"min": Xmin, "max": Xmax} as a dict
if isinstance(global_minmax, tuple) or isinstance(global_minmax, list):
    global_min, global_max = global_minmax[0], global_minmax[1]
elif isinstance(global_minmax, dict):
    # accept both "min"/"max" and "Xmin"/"Xmax" keys just in case
    if "min" in global_minmax and "max" in global_minmax:
        global_min, global_max = global_minmax["min"], global_minmax["max"]
    elif "Xmin" in global_minmax and "Xmax" in global_minmax:
        global_min, global_max = global_minmax["Xmin"], global_minmax["Xmax"]
    else:
        raise ValueError("global_minmax dict missing expected keys ('min'/'max' or 'Xmin'/'Xmax')")
else:
    # unexpected format
    raise ValueError("global_minmax has unexpected type: %s" % type(global_minmax))

st.sidebar.success("Models Loaded Successfully ✔")

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def plot_series(y, title, color="blue"):
    fig = plt.figure(figsize=(12,4))
    plt.plot(y, color=color)
    plt.title(title)
    plt.grid(True)
    st.pyplot(fig)

def global_minmax_scale(data):
    return (data - global_min) / (global_max - global_min + 1e-8)

def create_windows(data, window=100):
    X = []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
    return np.array(X)

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------

tab1, tab2 = st.tabs(["📊 Dashboard", "📁 Upload & Predict"])

# ============================================================
# TAB 1 — DASHBOARD VISUALIZATION
# ============================================================
with tab1:

    st.subheader("📊 Machine Health Overview")

    dashboard_df = pd.read_csv("Advanced_Hybrid_ML_Project/data/raw/dashboard_dataset.csv")

    current_health = dashboard_df["health_index"].iloc[-1]

    if current_health > 0.7:
        color = "green"; status = "HEALTHY"
    elif current_health > 0.4:
        color = "orange"; status = "WARNING"
    else:
        color = "red"; status = "CRITICAL FAILURE"

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

            raw = user_df[required_cols].values

            scaled = global_minmax_scale(raw)

            X = create_windows(scaled, 100)

            # AE anomaly
            X_rec = autoencoder.predict(X)
            anomaly_scores = np.mean((X - X_rec)**2, axis=(1,2))

            # LSTM RUL
            rul_scaled = lstm_rul.predict(X).flatten()
            rul_pred = rul_scaled * len(dashboard_df)

            # Fusion Model
            fusion_features = np.column_stack([anomaly_scores, rul_pred])
            fusion_scaled = fusion_scaler.transform(fusion_features)
            failure_prob = fusion_model.predict_proba(fusion_scaled)[:, 1]

            # Health Index
            anomaly_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
            health_index = 0.5 * (1 - anomaly_norm) + 0.5 * rul_scaled

            st.subheader("🔍 Prediction Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Health Index", f"{health_index[-1]:.2f}")
            col2.metric("Failure Probability", f"{failure_prob[-1]:.2f}")
            col3.metric("Predicted RUL", f"{rul_pred[-1]:.0f} steps")

            st.markdown("---")

            plot_series(health_index, "Health Index (Uploaded Data)", color="green")
            plot_series(failure_prob, "Failure Probability (Uploaded Data)", color="red")
            plot_series(anomaly_scores, "Anomaly Score (Uploaded Data)", color="orange")
            plot_series(rul_pred, "RUL Prediction (Uploaded Data)", color="purple")

