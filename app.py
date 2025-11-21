# app.py — Final fixed production-ready dashboard (drop-in replacement)
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
st.set_page_config(page_title="Hybrid ML Machine Health Dashboard",
                   layout="wide", page_icon="🛠️")
st.title("🛠️ Hybrid Machine Health Monitoring Dashboard")

# -----------------------------
# Paths & constants (adjust if needed)
# -----------------------------
BASE = Path(".")
MODELS_DIR = BASE / "Advanced_Hybrid_ML_Project" / "models"
DATA_RAW_DIR = BASE / "Advanced_Hybrid_ML_Project" / "data" / "raw"

AE_PATH = MODELS_DIR / "autoencoder_model"
LSTM_PATH = MODELS_DIR / "lstm_rul_model"
FUSION_MODEL_PATH = MODELS_DIR / "fusion_model_joblib.pkl"
FUSION_SCALER_PATH = MODELS_DIR / "fusion_feature_scaler_joblib.pkl"

# canonical scaler filename you confirmed
SCALER_PATH = MODELS_DIR / "scaler-1.pkl"

ANOMALY_STATS_PATH = MODELS_DIR / "anomaly_stats_joblib.pkl"  # optional
DASHBOARD_CSV = DATA_RAW_DIR / "dashboard_dataset.csv"

WINDOW = 100              # confirmed window size
MAX_RUL = 9900            # training convention used in your pipeline

# -----------------------------
# Small helpers
# -----------------------------
def st_error_and_stop(msg):
    st.error(msg)
    st.stop()

def plot_series(y, title, color="blue"):
    fig = plt.figure(figsize=(12, 3))
    plt.plot(y, color=color)
    plt.title(title)
    plt.grid(True)
    st.pyplot(fig)

def colored_badge(text, color):
    return f"<span style='background-color:{color}; padding:6px 12px; border-radius:10px; color:white; font-weight:700;'>{text}</span>"

# -----------------------------
# Load models & artifacts
# -----------------------------
st.sidebar.info("Loading models & artifacts...")

# TensorFlow models
try:
    autoencoder = tf.keras.models.load_model(str(AE_PATH))
    lstm_rul = tf.keras.models.load_model(str(LSTM_PATH))
except Exception as e:
    st_error_and_stop(f"Failed to load TF models (autoencoder/lstm). Error: {e}")

# Fusion artifacts
try:
    fusion_model = joblib.load(str(FUSION_MODEL_PATH))
    fusion_scaler = joblib.load(str(FUSION_SCALER_PATH))
except Exception as e:
    st_error_and_stop(f"Failed to load fusion artifacts. Error: {e}")

# Scaler (canonical — user confirmed scaler-1.pkl)
try:
    scaler = joblib.load(str(SCALER_PATH))
except Exception as e:
    st_error_and_stop(f"Failed to load the canonical scaler at {SCALER_PATH}. Error: {e}")

# Optional anomaly stats saved during training (min/max or mean/std)
anomaly_stats = None
if ANOMALY_STATS_PATH.exists():
    try:
        anomaly_stats = joblib.load(str(ANOMALY_STATS_PATH))
    except Exception:
        anomaly_stats = None

st.sidebar.success("Models & scaler loaded ✔")

# -----------------------------
# Load dashboard dataset if present (used for fallback stats)
# -----------------------------
if DASHBOARD_CSV.exists():
    try:
        dashboard_df = pd.read_csv(DASHBOARD_CSV)
    except Exception:
        dashboard_df = pd.DataFrame()
else:
    dashboard_df = pd.DataFrame()

# -----------------------------
# Preprocessing helpers
# -----------------------------
def create_windows_from_array(X: np.ndarray, window: int = WINDOW):
    n = X.shape[0]
    if n < window:
        pad_count = window - n + 1
        pad = np.repeat(X[-1:].reshape(1, -1), pad_count, axis=0)
        X = np.vstack([X, pad])
        n = X.shape[0]
    windows = np.stack([X[i:i+window] for i in range(n - window + 1)], axis=0)
    return windows

# -----------------------------
# UI: Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📊 Dashboard", "📁 Upload & Predict"])

# ---------- TAB 1: Dashboard view ----------
with tab1:
    st.subheader("📊 Machine Health Overview")
    if dashboard_df.empty:
        st.info("No dashboard dataset available. Use the Upload tab to test predictions.")
    else:
        def safe_get(col, default=0.0):
            return dashboard_df[col].iloc[-1] if col in dashboard_df.columns else default

        current_health = safe_get("health_index", 0.0)
        current_fp = safe_get("failure_probability", 0.0)
        current_rul = safe_get("rul_prediction", 0.0)

        if current_health > 0.7:
            status_color, status_text = "green", "HEALTHY"
        elif current_health > 0.4:
            status_color, status_text = "orange", "WARNING"
        else:
            status_color, status_text = "red", "CRITICAL FAILURE"

        st.markdown(f"<div style='padding:12px;border-radius:10px;background:#f5f5f5;text-align:center;'><h2 style='color:{status_color};'>Machine Status: {status_text}</h2></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Health Index", f"{current_health:.2f}")
        c2.metric("Failure Probability", f"{current_fp:.2f}")
        c3.metric("Predicted RUL", f"{current_rul:.0f} steps")

        st.markdown("---")
        option = st.radio("Select Visualization:", ["Health Index", "Failure Probability", "RUL Prediction", "Anomaly Score", "Raw Sensors"], horizontal=True)

        if option == "Health Index":
            if "health_index" in dashboard_df.columns:
                plot_series(dashboard_df["health_index"], "Health Index Over Time", color="green")
            else:
                st.info("health_index column not found in dashboard dataset")
        elif option == "Failure Probability":
            if "failure_probability" in dashboard_df.columns:
                plot_series(dashboard_df["failure_probability"], "Failure Probability Over Time", color="red")
            else:
                st.info("failure_probability column not found in dashboard dataset")
        elif option == "RUL Prediction":
            if "rul_prediction" in dashboard_df.columns:
                plot_series(dashboard_df["rul_prediction"], "RUL Prediction Over Time", color="purple")
            else:
                st.info("rul_prediction column not found in dashboard dataset")
        elif option == "Anomaly Score":
            if "anomaly_score" in dashboard_df.columns:
                plot_series(dashboard_df["anomaly_score"], "Anomaly Score Timeline", color="orange")
            else:
                st.info("anomaly_score column not found in dashboard dataset")
        else:
            sensor = st.selectbox("Choose sensor:", ["vibration","temperature","pressure","torque","current","rpm"])
            if sensor in dashboard_df.columns:
                plot_series(dashboard_df[sensor], f"{sensor.capitalize()} Over Time", color="blue")
            else:
                st.info(f"{sensor} not found in dashboard dataset")

# ---------- TAB 2: Upload & Predict ----------
with tab2:
    st.subheader("📁 Upload Sensor CSV for Real-Time Prediction")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is None:
        st.info("Upload CSV with columns: vibration,temperature,pressure,torque,current,rpm")
    else:
        # Read and validate
        try:
            user_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st_error_and_stop(f"Failed to read uploaded CSV: {e}")

        st.write(user_df.head())
        required_cols = ["vibration","temperature","pressure","torque","current","rpm"]
        if not set(required_cols).issubset(user_df.columns):
            st.error("CSV missing required sensor columns.")
        else:
            raw = user_df[required_cols].values.astype(float)

            # -------------------------
            # 1) SCALE using the canonical scaler (scaler-1.pkl) — EXACT training scaler
            # -------------------------
            try:
                scaled = scaler.transform(raw)    # this is the critical fix
            except Exception as e:
                st_error_and_stop(f"Failed to transform uploaded data with scaler. Error: {e}")

            # -------------------------
            # 2) WINDOW creation
            # -------------------------
            Xw = create_windows_from_array(scaled, WINDOW)
            n_windows = Xw.shape[0]
            train_n_windows = (dashboard_df.shape[0] - WINDOW + 1) if not dashboard_df.empty else max(1, n_windows)

            # -------------------------
            # 3) Autoencoder inference -> anomaly scores (raw MSE)
            # -------------------------
            try:
                X_rec = autoencoder.predict(Xw, verbose=0)
                anomaly_scores = np.mean((Xw - X_rec)**2, axis=(1,2))
            except Exception as e:
                st_error_and_stop(f"Autoencoder inference error: {e}")

            # -------------------------
            # 4) LSTM RUL inference
            # -------------------------
            try:
                lstm_out = lstm_rul.predict(Xw, verbose=0).flatten()
                if np.nanmax(lstm_out) <= 1.01:
                    rul_abs = lstm_out * MAX_RUL
                    rul_scaled_for_index = lstm_out
                else:
                    rul_abs = lstm_out
                    rul_scaled_for_index = np.clip(rul_abs / MAX_RUL, 0.0, 1.0)
                rul_abs = np.maximum(rul_abs, 0.0)
            except Exception as e:
                st_error_and_stop(f"LSTM inference error: {e}")

            # -------------------------
            # 5) Fusion prediction (use last window)
            # -------------------------
            latest_anom = float(anomaly_scores[-1])
            latest_rul_abs = float(rul_abs[-1])
            fusion_features = np.array([[latest_anom, latest_rul_abs]], dtype=float)
            try:
                fusion_scaled = fusion_scaler.transform(fusion_features)
                if hasattr(fusion_model, "predict_proba"):
                    failure_prob = float(fusion_model.predict_proba(fusion_scaled)[0][1])
                else:
                    failure_prob = float(np.clip(fusion_model.predict(fusion_scaled)[0], 0.0, 1.0))
            except Exception as e:
                st_error_and_stop(f"Fusion inference error: {e}")

            # -------------------------
            # 6) Anomaly normalization for health index (prefer training stats)
            # -------------------------
            an_min = None; an_max = None
            if anomaly_stats is not None and isinstance(anomaly_stats, dict):
                if "min" in anomaly_stats and "max" in anomaly_stats:
                    an_min, an_max = float(anomaly_stats["min"]), float(anomaly_stats["max"])
                elif "mean" in anomaly_stats and "std" in anomaly_stats:
                    an_min = float(anomaly_stats["mean"] - 3.0 * anomaly_stats["std"])
                    an_max = float(anomaly_stats["mean"] + 3.0 * anomaly_stats["std"])
            if an_min is None or an_max is None:
                if "anomaly_score" in dashboard_df.columns:
                    an_min = float(dashboard_df["anomaly_score"].min())
                    an_max = float(dashboard_df["anomaly_score"].max())
            if an_min is None or an_max is None:
                an_min = float(np.min(anomaly_scores))
                an_max = float(np.max(anomaly_scores))

            if (an_max - an_min) < 1e-9:
                an_norm_latest = 0.0
            else:
                an_norm_latest = (latest_anom - an_min) / (an_max - an_min)
                an_norm_latest = float(np.clip(an_norm_latest, 0.0, 1.0))

            # -------------------------
            # 7) Health index (simple thresholds chosen)
            # -------------------------
            rul_for_index = float(rul_scaled_for_index[-1]) if isinstance(rul_scaled_for_index, np.ndarray) else float(rul_scaled_for_index)
            health_index = 0.5 * (1.0 - an_norm_latest) + 0.5 * rul_for_index
            health_index = float(np.clip(health_index, 0.0, 1.0))

            # -------------------------
            # 8) Anomaly boolean & machine status
            # -------------------------
            an_mean = float(np.mean(anomaly_scores))
            an_std = float(np.std(anomaly_scores)) + 1e-9
            is_anomaly_local = latest_anom > (an_mean + 2.0 * an_std)
            is_anomaly_fusion = failure_prob > 0.5
            is_anomaly = bool(is_anomaly_local or is_anomaly_fusion)

            if health_index > 0.7:
                machine_status, status_color = "HEALTHY", "green"
            elif health_index > 0.4:
                machine_status, status_color = "WARNING", "orange"
            else:
                machine_status, status_color = "CRITICAL FAILURE", "red"

            if latest_rul_abs > 3000:
                rul_state, rul_color = "Long life remaining", "green"
            elif latest_rul_abs > 1000:
                rul_state, rul_color = "Mid-life (Degrading)", "orange"
            else:
                rul_state, rul_color = "End-of-Life Soon", "red"

            # -------------------------
            # 9) Display
            # -------------------------
            st.subheader("🔍 Prediction Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Health Index", f"{health_index:.2f}")
            c1.markdown(colored_badge(
                "Healthy (0.7–1.0)" if health_index>0.7 else
                "Warning (0.4–0.7)" if health_index>0.4 else
                "Critical (0–0.4)",
                status_color
            ), unsafe_allow_html=True)

            c2.metric("Failure Probability", f"{failure_prob:.2f}")
            c2.markdown(colored_badge(
                "Safe (<0.3)" if failure_prob < 0.3 else
                "Risky (0.3–0.7)" if failure_prob < 0.7 else
                "High Risk (>0.7)",
                "green" if failure_prob < 0.3 else "orange" if failure_prob < 0.7 else "red"
            ), unsafe_allow_html=True)

            c3.metric("Predicted RUL", f"{latest_rul_abs:.0f} steps")
            c3.markdown(colored_badge(rul_state, rul_color), unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"### ⚠️ Anomaly Detected: {colored_badge('TRUE','red') if is_anomaly else colored_badge('FALSE','green')}", unsafe_allow_html=True)
            st.markdown(f"### 🏭 Machine Status: {colored_badge(machine_status, status_color)}", unsafe_allow_html=True)
            st.markdown(f"### ⏳ RUL Interpretation: {colored_badge(rul_state, rul_color)}", unsafe_allow_html=True)

            st.markdown("---")
            plot_series(anomaly_scores, "Anomaly Score (Uploaded Data)", color="orange")
            plot_series(rul_abs, "RUL Predictions Over Time (Uploaded Data)", color="purple")
            plot_series([health_index], "Health Index (Uploaded Data)", color="green")
            plot_series([failure_prob], "Failure Probability (Uploaded Data)", color="red")

            # Debug expander
            with st.expander("Debug info (values & shapes)"):
                st.write({
                    "latest_anom": latest_anom,
                    "an_min_train_like": an_min,
                    "an_max_train_like": an_max,
                    "an_mean_uploaded": an_mean,
                    "an_std_uploaded": an_std,
                    "an_norm_latest": an_norm_latest,
                    "latest_rul_abs": latest_rul_abs,
                    "rul_scaled_for_index_last": float(rul_for_index),
                    "health_index": health_index,
                    "failure_prob": failure_prob,
                    "is_anomaly_local": is_anomaly_local,
                    "is_anomaly_fusion": is_anomaly_fusion
                })
