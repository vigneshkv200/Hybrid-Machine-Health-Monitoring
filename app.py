# app.py — FULL FIXED (drop-in replacement)
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
# Paths
# -----------------------------
BASE = Path(".")
MODELS_DIR = BASE / "Advanced_Hybrid_ML_Project" / "models"
AE_PATH = MODELS_DIR / "autoencoder_model"
LSTM_PATH = MODELS_DIR / "lstm_rul_model"
FUSION_MODEL_PATH = MODELS_DIR / "fusion_model_joblib.pkl"
FUSION_SCALER_PATH = MODELS_DIR / "fusion_feature_scaler_joblib.pkl"
GLOBAL_MINMAX_PATH = MODELS_DIR / "global_minmax_joblib.pkl"
ANOMALY_STATS_PATH = MODELS_DIR / "anomaly_stats_joblib.pkl"   # optional (min/max or mean/std saved during training)
DASHBOARD_CSV = BASE / "Advanced_Hybrid_ML_Project" / "data" / "raw" / "dashboard_dataset.csv"

WINDOW = 100
MAX_RUL = 9900  # training used 9900

# -----------------------------
# Helpers
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
    return f"""<span style="background-color:{color}; padding:6px 12px; border-radius:10px; color:white; font-weight:700;">{text}</span>"""

# -----------------------------
# Load models & artifacts
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
    st_error_and_stop(f"Failed to load fusion artifacts: {e}")

# global min/max (for raw sensor scaling)
try:
    gm = joblib.load(str(GLOBAL_MINMAX_PATH))
    if isinstance(gm, (list, tuple)):
        Xmin, Xmax = np.array(gm[0], dtype=float), np.array(gm[1], dtype=float)
    elif isinstance(gm, dict):
        # accept 'min'/'max' or 'Xmin'/'Xmax'
        if "min" in gm and "max" in gm:
            Xmin, Xmax = np.array(gm["min"], dtype=float), np.array(gm["max"], dtype=float)
        elif "Xmin" in gm and "Xmax" in gm:
            Xmin, Xmax = np.array(gm["Xmin"], dtype=float), np.array(gm["Xmax"], dtype=float)
        else:
            raise ValueError("global_minmax dict missing expected keys")
    else:
        raise ValueError("global_minmax has unexpected format")
except Exception as e:
    st_error_and_stop(f"Failed to load global_minmax: {e}")

# optional: anomaly stats saved during training (min/max or mean/std)
anomaly_stats = None
if ANOMALY_STATS_PATH.exists():
    try:
        anomaly_stats = joblib.load(str(ANOMALY_STATS_PATH))
    except Exception:
        anomaly_stats = None

st.sidebar.success("Models & artifacts loaded")

# -----------------------------
# Load dashboard dataset if present (used as fallback)
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
def scale_raw_sensors(X_raw: np.ndarray):
    denom = (Xmax - Xmin) + 1e-9
    return (X_raw - Xmin) / denom

def create_windows(X: np.ndarray, window: int = WINDOW):
    n = X.shape[0]
    if n < window:
        pad_count = window - n + 1
        pad = np.repeat(X[-1:].reshape(1, -1), pad_count, axis=0)
        X = np.vstack([X, pad])
        n = X.shape[0]
    return np.stack([X[i:i+window] for i in range(n - window + 1)], axis=0)

# -----------------------------
# Tabs UI
# -----------------------------
tab1, tab2 = st.tabs(["📊 Dashboard", "📁 Upload & Predict"])

# Tab 1: Dashboard
with tab1:
    st.subheader("📊 Machine Health Overview")
    if dashboard_df.empty:
        st.info("No dashboard dataset found. Use Upload tab to test.")
    else:
        def safe(col, d=0.0):
            return dashboard_df[col].iloc[-1] if col in dashboard_df.columns else d
        cur_hi = safe("health_index", 0.0)
        cur_fp = safe("failure_probability", 0.0)
        cur_rul = safe("rul_prediction", 0.0)

        if cur_hi > 0.7:
            st.markdown(colored_badge("HEALTHY", "green"), unsafe_allow_html=True)
        elif cur_hi > 0.4:
            st.markdown(colored_badge("WARNING", "orange"), unsafe_allow_html=True)
        else:
            st.markdown(colored_badge("CRITICAL FAILURE", "red"), unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Health Index", f"{cur_hi:.2f}")
        col2.metric("Failure Probability", f"{cur_fp:.2f}")
        col3.metric("Predicted RUL", f"{cur_rul:.0f} steps")
        st.markdown("---")
        option = st.radio("Visualization:", ["Health Index", "Failure Probability", "RUL Prediction", "Anomaly Score", "Raw Sensors"], horizontal=True)
        if option == "Health Index" and "health_index" in dashboard_df:
            plot_series(dashboard_df["health_index"], "Health Index", color="green")
        elif option == "Failure Probability" and "failure_probability" in dashboard_df:
            plot_series(dashboard_df["failure_probability"], "Failure Probability", color="red")
        elif option == "RUL Prediction" and "rul_prediction" in dashboard_df:
            plot_series(dashboard_df["rul_prediction"], "RUL Prediction", color="purple")
        elif option == "Anomaly Score" and "anomaly_score" in dashboard_df:
            plot_series(dashboard_df["anomaly_score"], "Anomaly Score", color="orange")
        elif option == "Raw Sensors":
            sensor = st.selectbox("Choose sensor:", ["vibration","temperature","pressure","torque","current","rpm"])
            if sensor in dashboard_df:
                plot_series(dashboard_df[sensor], f"{sensor} over time", color="blue")

# Tab 2: Upload & Predict (core fixes)
with tab2:
    st.subheader("📁 Upload Sensor CSV for Real-Time Prediction")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is None:
        st.info("Upload CSV with columns: vibration,temperature,pressure,torque,current,rpm")
    else:
        try:
            user_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st_error_and_stop(f"Failed to read uploaded CSV: {e}")

        st.write(user_df.head())
        required = ["vibration","temperature","pressure","torque","current","rpm"]
        if not set(required).issubset(user_df.columns):
            st.error("CSV missing required columns")
        else:
            raw = user_df[required].values.astype(float)

            # 1) Scale raw sensor values using training global min/max (same as training pipeline)
            scaled = scale_raw_sensors(raw)

            # 2) Create windows
            Xw = create_windows(scaled, WINDOW)
            n_windows = Xw.shape[0]
            train_n_windows = (dashboard_df.shape[0] - WINDOW + 1) if not dashboard_df.empty else max(1, n_windows)

            # 3) Autoencoder inference -> anomaly scores (raw MSE)
            try:
                X_rec = autoencoder.predict(Xw, verbose=0)
                anomaly_scores = np.mean((Xw - X_rec)**2, axis=(1,2))
            except Exception as e:
                st_error_and_stop(f"Autoencoder inference error: {e}")

            # 4) LSTM RUL inference -> handle normalized vs absolute
            try:
                lstm_out = lstm_rul.predict(Xw, verbose=0).flatten()
                # If model outputs are normalized (<=1.01), map to MAX_RUL
                if np.nanmax(lstm_out) <= 1.01:
                    rul_abs = lstm_out * MAX_RUL
                    rul_scaled_for_index = lstm_out  # in [0,1]
                else:
                    # model output is absolute RUL already
                    rul_abs = lstm_out
                    rul_scaled_for_index = np.clip(rul_abs / MAX_RUL, 0.0, 1.0)
                rul_abs = np.maximum(rul_abs, 0.0)
            except Exception as e:
                st_error_and_stop(f"LSTM inference error: {e}")

            # 5) Prepare fusion features in the same space used during training
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

            # 6) Stable anomaly normalization for health index
            # Preference order for training anomaly stats:
            #  a) loaded anomaly_stats (if saved during training)
            #  b) dashboard_df anomaly_score min/max (if available)
            #  c) fallback to anomaly_scores min/max from current uploaded windows (last resort)
            an_min = None; an_max = None
            if anomaly_stats is not None:
                # anomaly_stats may be dict with 'min'/'max' or 'mean'/'std'
                if isinstance(anomaly_stats, dict) and "min" in anomaly_stats and "max" in anomaly_stats:
                    an_min, an_max = float(anomaly_stats["min"]), float(anomaly_stats["max"])
                elif isinstance(anomaly_stats, dict) and "mean" in anomaly_stats and "std" in anomaly_stats:
                    # convert to approximate min/max = mean +/- 3*std
                    an_min = float(anomaly_stats["mean"] - 3.0 * anomaly_stats["std"])
                    an_max = float(anomaly_stats["mean"] + 3.0 * anomaly_stats["std"])
            if an_min is None or an_max is None:
                if "anomaly_score" in dashboard_df.columns:
                    an_min = float(dashboard_df["anomaly_score"].min())
                    an_max = float(dashboard_df["anomaly_score"].max())
            if an_min is None or an_max is None:
                # fallback to uploaded anomaly window stats (not ideal but safe)
                an_min = float(np.min(anomaly_scores))
                an_max = float(np.max(anomaly_scores))
            # avoid division by zero
            if (an_max - an_min) < 1e-9:
                an_norm_latest = 0.0
            else:
                an_norm_latest = (latest_anom - an_min) / (an_max - an_min)
                an_norm_latest = float(np.clip(an_norm_latest, 0.0, 1.0))

            # 7) Health index: combine anomaly (lower better) and normalized RUL (higher better)
            hi = 0.5 * (1.0 - an_norm_latest) + 0.5 * rul_scaled_for_index[-1] if isinstance(rul_scaled_for_index, np.ndarray) else 0.5 * (1.0 - an_norm_latest) + 0.5 * float(rul_scaled_for_index)
            hi = float(np.clip(hi, 0.0, 1.0))

            # 8) Anomaly boolean (two-pronged): local z-score + fusion probability
            an_mean = float(np.mean(anomaly_scores))
            an_std = float(np.std(anomaly_scores)) + 1e-9
            is_anomaly_local = latest_anom > (an_mean + 2.0 * an_std)
            is_anomaly_fusion = failure_prob > 0.5
            is_anomaly = bool(is_anomaly_local or is_anomaly_fusion)

            # 9) Machine status & RUL interpretation
            if hi > 0.7:
                machine_status, status_color = "HEALTHY", "green"
            elif hi > 0.4:
                machine_status, status_color = "WARNING", "orange"
            else:
                machine_status, status_color = "CRITICAL FAILURE", "red"

            if latest_rul_abs > 3000:
                rul_state, rul_color = "Long life remaining", "green"
            elif latest_rul_abs > 1000:
                rul_state, rul_color = "Mid-life (Degrading)", "orange"
            else:
                rul_state, rul_color = "End-of-Life Soon", "red"

            # -----------------------------
            # Display results (with proper HTML rendering)
            # -----------------------------
            st.subheader("🔍 Prediction Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Health Index", f"{hi:.2f}")
            c1.markdown(colored_badge(
                "Healthy (0.7–1.0)" if hi>0.7 else
                "Warning (0.4–0.7)" if hi>0.4 else
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
            # Plots
            plot_series(anomaly_scores, "Anomaly Score (Uploaded Data)", color="orange")
            plot_series(rul_abs, "RUL Predictions Over Time (Uploaded Data)", color="purple")
            plot_series([hi], "Health Index (Uploaded Data)", color="green")
            plot_series([failure_prob], "Failure Probability (Uploaded Data)", color="red")

            # Debug panel
            with st.expander("Debug info (values & shapes)"):
                st.write({
                    "latest_anom": latest_anom,
                    "an_min_train_like": an_min,
                    "an_max_train_like": an_max,
                    "an_mean_uploaded": an_mean,
                    "an_std_uploaded": an_std,
                    "an_norm_latest": an_norm_latest,
                    "latest_rul_abs": latest_rul_abs,
                    "rul_scaled_for_index_last": float(rul_scaled_for_index[-1]) if isinstance(rul_scaled_for_index, np.ndarray) else float(rul_scaled_for_index),
                    "health_index": hi,
                    "failure_prob": failure_prob,
                    "is_anomaly_local": is_anomaly_local,
                    "is_anomaly_fusion": is_anomaly_fusion
                })
