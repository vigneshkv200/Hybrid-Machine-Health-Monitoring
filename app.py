# app.py — Final production-ready (copy entire file)
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
# Paths (adjust if you moved files)
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
    return f"""
    <span style="background-color:{color};
                 padding:6px 12px;
                 border-radius:10px;
                 color:white;
                 font-weight:700;">
        {text}
    </span>
    """

# -----------------------------
# Load models (robust)
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
    st_error_and_stop(f"Failed to load fusion joblib artifacts: {e}")

# global min/max may be tuple (Xmin, Xmax) or dict
try:
    gm = joblib.load(str(GLOBAL_MINMAX_PATH))
    if isinstance(gm, (tuple, list)):
        Xmin, Xmax = np.array(gm[0], dtype=float), np.array(gm[1], dtype=float)
    elif isinstance(gm, dict):
        if "min" in gm and "max" in gm:
            Xmin, Xmax = np.array(gm["min"], dtype=float), np.array(gm["max"], dtype=float)
        elif "Xmin" in gm and "Xmax" in gm:
            Xmin, Xmax = np.array(gm["Xmin"], dtype=float), np.array(gm["Xmax"], dtype=float)
        else:
            raise ValueError("global_minmax dict missing expected keys")
    else:
        raise ValueError("global_minmax has unexpected type")
except Exception as e:
    st_error_and_stop(f"Failed to load global min/max: {e}")

st.sidebar.success("Models loaded")

# -----------------------------
# Load dashboard dataset (optional)
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
    # elementwise min-max
    denom = (Xmax - Xmin) + 1e-9
    return (X_raw - Xmin) / denom

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
# UI tabs
# -----------------------------
tab1, tab2 = st.tabs(["📊 Dashboard", "📁 Upload & Predict"])

# ---------- TAB 1 ----------
with tab1:
    st.subheader("📊 Machine Health Overview")
    if dashboard_df.empty:
        st.info("No dashboard dataset available. Use the Upload tab to test.")
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

        col1, col2, col3 = st.columns(3)
        col1.metric("Health Index", f"{current_health:.2f}")
        col2.metric("Failure Probability", f"{current_fp:.2f}")
        col3.metric("Predicted RUL", f"{current_rul:.0f} steps")

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
        elif option == "Raw Sensors":
            sensor = st.selectbox("Choose Sensor:", ["vibration","temperature","pressure","torque","current","rpm"])
            if sensor in dashboard_df.columns:
                plot_series(dashboard_df[sensor], f"{sensor.capitalize()} Over Time", color="blue")
            else:
                st.info(f"{sensor} not found in dashboard dataset")

# ---------- TAB 2 ----------
with tab2:
    st.subheader("📁 Upload Sensor CSV for Real-Time Prediction")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a sensor CSV with columns: vibration, temperature, pressure, torque, current, rpm")
    else:
        try:
            user_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st_error_and_stop(f"Failed to read uploaded CSV: {e}")

        st.write(user_df.head())
        required_cols = ["vibration","temperature","pressure","torque","current","rpm"]
        if not set(required_cols).issubset(user_df.columns):
            st.error("CSV missing required sensor columns")
        else:
            raw = user_df[required_cols].values.astype(float)
            # scale using training global min/max
            scaled = scale_raw_sensors(raw)
            Xw = create_windows_from_array(scaled, WINDOW)
            n_windows = Xw.shape[0]
            # reference train window count if dashboard available
            train_n_windows = (dashboard_df.shape[0] - WINDOW + 1) if not dashboard_df.empty else max(1, n_windows)

            # AE inference
            try:
                X_rec = autoencoder.predict(Xw, verbose=0)
                anomaly_scores = np.mean((Xw - X_rec)**2, axis=(1,2))
            except Exception as e:
                st_error_and_stop(f"Autoencoder inference error: {e}")

            # LSTM RUL inference
            try:
                lstm_out = lstm_rul.predict(Xw, verbose=0).flatten()
                # If model outputs in [0,1], scale to train window count
                if np.max(lstm_out) <= 1.01:
                    rul_vals = lstm_out * float(train_n_windows)
                else:
                    rul_vals = lstm_out.copy()
                # ensure non-negative
                rul_vals = np.maximum(rul_vals, 0.0)
            except Exception as e:
                st_error_and_stop(f"LSTM inference error: {e}")

            # fusion prediction (use last window)
            latest_anom = float(anomaly_scores[-1])
            latest_rul = float(rul_vals[-1])
            fusion_features = np.array([[latest_anom, latest_rul]], dtype=float)
            try:
                fusion_scaled = fusion_scaler.transform(fusion_features)
                if hasattr(fusion_model, "predict_proba"):
                    failure_prob = float(fusion_model.predict_proba(fusion_scaled)[0][1])
                else:
                    failure_prob = float(np.clip(fusion_model.predict(fusion_scaled)[0], 0.0, 1.0))
            except Exception as e:
                st_error_and_stop(f"Fusion inference error: {e}")

            # anomaly boolean: anomaly if latest_anom >> distribution or fusion_prob high
            # use mean+2*std heuristic on anomaly_scores
            an_mean = float(np.mean(anomaly_scores))
            an_std = float(np.std(anomaly_scores)) + 1e-9
            is_anomaly_local = latest_anom > (an_mean + 2.0 * an_std)
            is_anomaly_fusion = failure_prob > 0.5
            is_anomaly = bool(is_anomaly_local or is_anomaly_fusion)

            # stable anomaly normalization for health index
            # use z-score clamp to 0-1
            an_z = (latest_anom - an_mean) / an_std
            an_norm = np.clip((an_z + 3) / 6.0, 0.0, 1.0)  # map z in [-3,3] -> [0,1]

            # RUL normalization for health index: normalized by train_n_windows
            rul_norm = float(np.clip(latest_rul / max(1.0, float(train_n_windows)), 0.0, 1.0))

            # Health index: combine anomaly (lower is better) and RUL (higher is better)
            health_index = 0.5 * (1.0 - an_norm) + 0.5 * rul_norm
            health_index = float(np.clip(health_index, 0.0, 1.0))

            # Determine machine status & RUL interpretation
            if health_index > 0.7:
                machine_status = "HEALTHY"; status_color = "green"
            elif health_index > 0.4:
                machine_status = "WARNING"; status_color = "orange"
            else:
                machine_status = "CRITICAL FAILURE"; status_color = "red"

            if latest_rul > 3000:
                rul_state = "Long life remaining"; rul_color = "green"
            elif latest_rul > 1000:
                rul_state = "Mid-life (Degrading)"; rul_color = "orange"
            else:
                rul_state = "End-of-Life Soon"; rul_color = "red"

            # Display metrics and badges
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

            c3.metric("Predicted RUL", f"{latest_rul:.0f} steps")
            c3.markdown(colored_badge(rul_state, rul_color), unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"### ⚠️ Anomaly Detected: {colored_badge('TRUE', 'red') if is_anomaly else colored_badge('FALSE','green')}", unsafe_allow_html=True)
            st.markdown(f"### 🏭 Machine Status: {colored_badge(machine_status, status_color)}", unsafe_allow_html=True)
            st.markdown(f"### ⏳ RUL Interpretation: {colored_badge(rul_state, rul_color)}", unsafe_allow_html=True)

            st.markdown("---")
            # Plots
            plot_series(anomaly_scores, "Anomaly Score (Uploaded Data)", color="orange")
            plot_series(rul_vals, "RUL Predictions Over Time (Uploaded Data)", color="purple")
            plot_series([health_index], "Health Index (Uploaded Data)", color="green")
            plot_series([failure_prob], "Failure Probability (Uploaded Data)", color="red")

            # small debug info (optional)
            with st.expander("Debug info (show values)"):
                st.write({
                    "latest_anomaly": latest_anom,
                    "anomaly_mean": an_mean,
                    "anomaly_std": an_std,
                    "an_norm": an_norm,
                    "latest_rul": latest_rul,
                    "rul_norm": rul_norm,
                    "health_index": health_index,
                    "failure_prob": failure_prob,
                    "is_anomaly_local": is_anomaly_local,
                    "is_anomaly_fusion": is_anomaly_fusion
                })
