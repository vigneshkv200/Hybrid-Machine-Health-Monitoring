# 🚀 Hybrid Machine Health Monitoring System

A full **industrial-grade predictive maintenance system** built using:

- **Autoencoder** (Unsupervised anomaly detection)
- **LSTM** (Remaining Useful Life prediction)
- **Fusion Classifier** (Combines anomaly + RUL for final failure risk)
- **Streamlit Dashboard** (Real-time machine health monitoring)
- **Complete Backend Pipeline** (Testing, validation, preprocessing)

This project provides **health index, failure probability, anomaly score, and RUL** from raw sensor data.

---

## 📌 1. Project Overview

This system predicts:

- Machine health status (Healthy → Warning → Critical)
- Failure probability
- Remaining Useful Life (RUL)
- Anomaly score using autoencoder
- Multisensor behavior visualization

The goal is to simulate a **real industry predictive maintenance pipeline** used in:

- CNC machines
- Motors & pumps
- HVAC systems
- Turbines
- EV motor diagnostics

---

## 📌 2. Features

### ✅ Real-time CSV Upload & Prediction

- Upload 6-sensor data
- Instant prediction of health metrics

### ✅ Hybrid ML Pipeline

- Autoencoder → detects anomalies
- LSTM → forecasts RUL using sequences
- Fusion Classifier → final failure risk

### ✅ Interactive Streamlit Dashboard

- Health Index Graph
- Failure Probability Graph
- Anomaly Score
- RUL Trends
- Raw sensor plots

### ✅ Backend Verification Notebook

Includes:

- Model loading
- Preprocessing
- Windowing
- Predictions
- Plots

---

## 📌 3. Folder Structure

```
Advanced_Hybrid_ML_Project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── autoencoder_model/
│   ├── lstm_rul_model/
│   ├── fusion_model.pkl
│   └── scaler.pkl
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│   └── backend_test.ipynb
│
└── README.md
```

---

## 📌 4. Tech Stack

- **Python**
- **TensorFlow==2.12.0 / Keras**
- **Scikit-Learn**
- **NumPy / Pandas**
- **Matplotlib / Seaborn**
- **Streamlit**

Install using:
```bash
pip install -r requirements.txt
```
---

## 📌 5. Data Used

The system uses **synthetic multisensor time-series data** including:

- Vibration
- Temperature
- Pressure
- Torque
- Current
- RPM

Healthy + Degrading + Failure regions are simulated.

---

## 📌 6. Models Used

### 🔹 **1. Autoencoder (Anomaly Detection)**

Learns healthy machine patterns → reconstruction error = anomaly score.

### 🔹 **2. LSTM (Remaining Useful Life)**

Predicts RUL using sliding-window sequences.

### 🔹 **3. Fusion Classifier**

Takes:

- Anomaly score
- RUL

Outputs:

- Failure probability
- Machine status

---

## 📌 7. Streamlit Dashboard Preview

**Includes:**

- Machine Status Card
- Health Index
- Failure Probability
- RUL
- Upload & Predict Page
- Interactive charts

(Insert screenshots here)

---

## 📌 8. How to Run the Project

### 🔹 1. Install dependencies

```
pip install -r requirements.txt
```

### 🔹 2. Run Streamlit app

```
cd dashboard
streamlit run app.py
```

### 🔹 3. Upload sensor CSV

System will automatically:

- Preprocess
- Create window sequences
- Predict anomaly, RUL, failure probability

---

## 📌 9. Example Prediction Output

```
Health Index: 0.34
Failure Probability: 1.00
Predicted RUL: 1675 steps
Machine Status: CRITICAL FAILURE
```

---

## 📌 10. Real-World Applications

- Predictive maintenance in factories
- Motor/pump health prediction
- Turbine monitoring
- Smart manufacturing systems
- Robotics sensor diagnostics
- Industrial IoT monitoring

---

## 📌 11. Future Improvements

- Add GRU-based RUL model
- Add sensor drift compensation
- Deploy using Docker + Render
- Add database logging
- Multi-machine monitoring

---

## 📌 12. Author

👤 **Vignesh KV**

- Final Year AI/ML Engineering Student 
- Passionate about ML Engineering & Industrial AI
- **LinkedIn:** https://www.linkedin.com/in/vigneshkv200
- **GitHub:** https://github.com/vigneshkv200


---

## ⭐ Final Note

This project demonstrates **real ML engineering**, not just model training:

- End-to-end pipeline
- Deployment-ready dashboard
- Industrial simulation
- Multimodal hybrid architecture


