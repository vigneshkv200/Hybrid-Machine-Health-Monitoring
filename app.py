# app.py (FINAL - robust + deploy-ready)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Hybrid ML Machine Health Dashboard",
                   layout="wide", page_icon="")
st.title(" Hybrid Machine Health Monitoring Dashboard")

# -----------------------------
# Paths (expecting models in Advanced_Hybrid_ML_Project/models)
# -----------------------------
BASE = Path(".")
AE_PATH = BASE / "Advanced_Hybrid_ML_Project" / "models" / "autoencoder_model"
LSTM_PATH = BASE / "Advanced_Hybrid_ML_Project" / "models" / "lstm_rul_model"
FUSION_MODEL_PATH = BASE / "Advanced_Hybrid_ML_Project" / "models" / "fusion_model_joblib.pkl"
FUSION_SCALER_PATH = BASE / "Advanced_Hybrid_ML_Project" / "models" / "fusion_feature_scaler_joblib.pkl"
GLOBAL_MINMAX_PATH = BASE / "Advanced_Hybrid_ML_Project" / "models" / "global_minmax_joblib.pkl"
DASHBOARD_CSV = BASE / "Advanced_Hybrid_ML_Project" / "data" / "raw" / "dashboard_dataset.csv"

# -----------------------------
# Utilities
# -----------------------------
def st_error_and_stop(msg):
    st.error(msg)
    st.stop()

def safe_joblib_load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return joblib.load(path)

def plot_series(y, title, color="blue"):
    fig = plt.figure(figsize=(12, 4))
    plt.plot(y, color=color)
    plt.title(title)
    plt.grid(True)
    st.pyplot(fig)

# -----------------------------
# Load models & scalers (robust)
# -----------------------------
try:
    # TensorFlow models
    autoencoder = tf.keras.models.load_model(str(AE_PATH))
    lstm_rul = tf.keras.models.load_model(str(LSTM_PATH))
except Exception as e:
    st_error_and_stop(f"Failed to load TensorFlow models: {e}")

# Load joblib artifacts
try:
    fusion_model = safe_joblib_load(FUSION_MODEL_PATH)
    fusion_scaler = safe_joblib_load(FUSION_SCALER_PATH)
except Exception as e:
    st_error_and_stop(f"Failed to load fusion joblib models: {e}")

# Load global min/max and support tuple/dict formats
try:
    global_minmax = safe_joblib_load(GLOBAL_MINMAX_PATH)
    if isinstance(global_minmax, (tuple, list)):
        Xmin, Xmax = global_minmax[0], global_minmax[1]
    elif isinstance(global_minmax, dict):
        # accept multiple key styles
        if "min" in global_minmax and "max" in global_minmax:
            Xmin, Xmax = np.array(global_minmax["min"]), np.array(global_minmax["max"])
        elif "Xmin" in global_minmax and "Xmax" in global_minmax:
            Xmin, Xmax = np.array(global_minmax["Xmin"]), np.array(global_minmax["Xmax"])
        else:
            raise ValueError("global_minmax dict missing expected keys ('min'/'max' or 'Xmin'/'Xmax')")
    else:
        raise ValueError("global_minmax has unexpected type: %s" % type(global_minmax))
    Xmin = np.array(Xmin, dtype=float)
    Xmax = np.array(Xmax, dtype=float)
except Exception as e:
    st_error_and_stop(f"Failed to load global_minmax joblib: {e}")

# -----------------------------
# Load dashboard dataset (for display + to compute training window counts)
# -----------------------------
if not DASHBOARD_CSV.exists():
    st.warning("dashboard_dataset.csv not found in repo. Dashboard visualizations will be limited.")
    dashboard_df = pd.DataFrame()
else:
    dashboard_df = pd.read_csv(DASHBOARD_CSV)

# -----------------------------
# Preprocessing helper (same scaling used during training)
# -----------------------------
WINDOW = 100

def create_windows_from_array(X: np.ndarray, window: int = WINDOW):
    # X is (n_samples, n_features)
    n = X.shape[0]
    if n < window:
        # pad by repeating last row to create at least one window
        pad_count = window - n + 1
        pad = np.repeat(X[-1:].reshape(1, -1), pad_count, axis=0)
        X = np.vstack([X, pad])
        n = X.shape[0]
    # sliding windows
    # sliding_window_view may be unavailable, use manual
    windows = np.stack([X[i:i+window] for i in range(n - window + 1)], axis=0)
    return windows  # shape (n_windows, window, features)

def scale_raw_sensors(X_raw: np.ndarray):
    # Xmin and Xmax are arrays of length n_features
    denom = (Xmax - Xmin) + 1e-9
    return (X_raw - Xmin) / denom

# -----------------------------
# UI Tabs
# -----------------------------
tab1, tab2 = st.tabs([" Dashboard", " Upload & Predict"])

# ---------- TAB 1: Dashboard ----------
with tab1:
    st.subheader(" Machine Health Overview")

    if dashboard_df.empty:
        st.info("No dashboard dataset found. Upload a CSV in the Upload tab to run predictions.")
    else:
        # show last computed metrics if present; otherwise safe defaults
        def safe_get(col, default=0.0):
            return dashboard_df[col].iloc[-1] if col in dashboard_df.columns else default

        current_health = safe_get("health_index", 0.0)
        current_fail_prob = safe_get("failure_probability", 0.0)
        current_rul = safe_get("rul_prediction", 0.0)

        if current_health > 0.7:
            color = "green"; status = "HEALTHY"
        elif current_health > 0.4:
            color = "orange"; status = "WARNING"
        else:
            color = "red"; status = "CRITICAL FAILURE"

        st.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#f0f0f0; text-align:center;'><h2 style='color:{color};'>Machine Status: {status}</h2></div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Health Index", f"{current_health:.2f}")
        col2.metric("Failure Probability", f"{current_fail_prob:.2f}")
        col3.metric("Predicted RUL", f"{current_rul:.0f} steps")

        st.markdown("---")
        option = st.radio("Select Visualization:", ["Health Index", "Failure Probability", "RUL Prediction", "Anomaly Score", "Raw Sensors"], horizontal=True)
        if option == "Health Index":
            if "health_index" in dashboard_df.columns:
                plot_series(dashboard_df["health_index"], "Health Index Over Time", color="green")
            else:
                st.info("health_index column not found in dashboard_dataset.csv")
        elif option == "Failure Probability":
            if "failure_probability" in dashboard_df.columns:
                plot_series(dashboard_df["failure_probability"], "Failure Probability Over Time", color="red")
            else:
                st.info("failure_probability column not found in dashboard_dataset.csv")
        elif option == "RUL Prediction":
            if "rul_prediction" in dashboard_df.columns:
                plot_series(dashboard_df["rul_prediction"], "RUL Prediction Over Time", color="purple")
            else:
                st.info("rul_prediction column not found in dashboard_dataset.csv")
        elif option == "Anomaly Score":
            if "anomaly_score" in dashboard_df.columns:
                plot_series(dashboard_df["anomaly_score"], "Anomaly Score Timeline", color="orange")
            else:
                st.info("anomaly_score column not found in dashboard_dataset.csv")
        elif option == "Raw Sensors":
            sensor = st.selectbox("Choose Sensor:", ["vibration", "temperature", "pressure", "torque", "current", "rpm"])
            if sensor in dashboard_df.columns:
                plot_series(dashboard_df[sensor], f"{sensor.capitalize()} Over Time", color="blue")
            else:
                st.info(f"{sensor} not found in dashboard dataset")

# ---------- TAB 2: Upload & Predict ----------
with tab2:
    st.subheader(" Upload Sensor CSV for Real-Time Prediction")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        try:
            user_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st_error_and_stop(f"Failed to read uploaded CSV: {e}")

        st.success("File uploaded successfully!")
        st.write(user_df.head())

        required_cols = ["vibration", "temperature", "pressure", "torque", "current", "rpm"]
        if not set(required_cols).issubset(user_df.columns):
            st.error(" CSV missing required sensor columns: vibration, temperature, pressure, torque, current, rpm")
        else:
            # --- Preprocess uploaded data ---
            raw = user_df[required_cols].values.astype(float)

            # Use the same global min/max scaling used during training
            try:
                scaled = scale_raw_sensors(raw)  # shape (n_samples, n_features)
            except Exception as e:
                st_error_and_stop(f"Scaling error: {e}")

            # create windows (n_windows, WINDOW, n_features)
            Xw = create_windows_from_array(scaled, WINDOW)
            n_windows = Xw.shape[0]
            # compute the training-time reference for max RUL
            # In training we used number_of_windows = len(dashboard_dataset) - WINDOW + 1
            train_n_windows = (dashboard_df.shape[0] - WINDOW + 1) if not dashboard_df.empty else n_windows

            # --- Autoencoder anomaly scores ---
            try:
                X_rec = autoencoder.predict(Xw, verbose=0)
                anomaly_scores = np.mean((Xw - X_rec) ** 2, axis=(1, 2))
            except Exception as e:
                st_error_and_stop(f"Autoencoder inference error: {e}")

            # --- LSTM RUL predictions ---
            try:
                lstm_out = lstm_rul.predict(Xw, verbose=0).flatten()
                # lstm_out should be in same scale used when training fusion.
                # If LSTM was trained to predict absolute window counts (like range n_windows->0),
                # it will already be OK. If scaled, adjust accordingly. We assume model predicts absolute RUL-like values.
                # Ensure non-negative:
                rul_pred_vals = np.maximum(lstm_out, 0.0)
                # If model produces values in [0,1], map to window count:
                # Heuristic: if max(lstm_out) <= 1.01 then treat as normalized and scale by train_n_windows
                if np.max(lstm_out) <= 1.01:
                    rul_pred_vals = lstm_out * float(train_n_windows)
                # pick last window's RUL as the current prediction
                rul_pred = float(rul_pred_vals[-1])
            except Exception as e:
                st_error_and_stop(f"LSTM inference error: {e}")

            # --- Fusion model prediction ---
            try:
                # use latest anomaly score & RUL
                latest_anom = float(anomaly_scores[-1])
                latest_rul = float(rul_pred)
                fusion_features = np.array([[latest_anom, latest_rul]], dtype=float)
                fusion_features_scaled = fusion_scaler.transform(fusion_features)
                if hasattr(fusion_model, "predict_proba"):
                    failure_probability = float(fusion_model.predict_proba(fusion_features_scaled)[0][1])
                else:
                    # fallback for models without predict_proba
                    failure_probability = float(fusion_model.predict(fusion_features_scaled)[0])
                    # map to [0,1]
                    failure_probability = float(np.clip(failure_probability, 0.0, 1.0))
            except Exception as e:
                st_error_and_stop(f"Fusion model inference error: {e}")

            # --- Health index (stable normalization) ---
            try:
                # normalize anomaly across windows to 0-1 (if constant, avoid div by zero)
                an_min = float(np.min(anomaly_scores))
                an_max = float(np.max(anomaly_scores))
                an_norm_last = 0.0
                if (an_max - an_min) > 1e-9:
                    an_norm_last = (latest_anom - an_min) / (an_max - an_min)
                    an_norm_last = float(np.clip(an_norm_last, 0.0, 1.0))
                # normalize RUL to [0,1] using train_n_windows as max possible
                rul_norm_last = float(np.clip(latest_rul / max(1.0, float(train_n_windows)), 0.0, 1.0))
                # health index: combine (higher RUL -> healthier) and (lower anomaly -> healthier)
                health_index = 0.5 * (1.0 - an_norm_last) + 0.5 * rul_norm_last
                health_index = float(np.clip(health_index, 0.0, 1.0))
            except Exception as e:
                st_error_and_stop(f"Health index computation error: {e}")

            # --- Display results ---
            st.subheader(" Prediction Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Health Index", f"{health_index:.2f}")
            col2.metric("Failure Probability", f"{failure_probability:.2f}")
            col3.metric("Predicted RUL", f"{rul_pred:.0f} steps")

            st.markdown("---")
            # Plots (show whole-window arrays)
            plot_series(anomaly_scores, "Anomaly Score (Uploaded Data)", color="orange")
            plot_series(rul_pred_vals, "RUL Prediction (Uploaded Data)", color="purple")
            # show scalar charts for single-value metrics
            plot_series([health_index], "Health Index (Uploaded Data)", color="green")
            plot_series([failure_probability], "Failure Probability (Uploaded Data)", color="red")
