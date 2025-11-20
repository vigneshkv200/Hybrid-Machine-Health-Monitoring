# 🚀 Hybrid Machine Health Monitoring System

A full **industrial‑grade predictive maintenance system** built using a hybrid Deep Learning + Classical ML architecture. This README is formatted in the **decorative, structured, modern style** exactly like your demo file — but rewritten fully for your project.

---

## ⭐ Overview
This project predicts **machine failures BEFORE they happen** using 6 real‑world sensor streams:
- Vibration
- Temperature
- Pressure
- Torque
- Current
- RPM

Using these, the system computes:
- **Health Index**
- **Anomaly Score** (Autoencoder)
- **Remaining Useful Life (RUL)** (LSTM)
- **Failure Probability** (Fusion Classifier)

This pipeline resembles how **smart factories / Industry 4.0** operate.

---

## 📂 Project Structure
```
Advanced_Hybrid_ML_Project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── test/
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

## 🔧 Tech Stack
- **Python 3.10**
- **TensorFlow / Keras** (Autoencoder + LSTM)
- **Scikit‑Learn** (Fusion Classifier + Scaling)
- **Streamlit** (Dashboard)
- **Pandas / NumPy**
- **Matplotlib / Seaborn**

---

## 📊 Data Used
Synthetic multisensor time‑series data representing:
- Normal operation
- Early degradation
- Critical near‑failure phase

Each sensor contributes unique failure patterns.

---

## 🧠 Models Used
### **1️⃣ Autoencoder – Anomaly Detection**
Learns only healthy data → When reconstruction error increases → anomaly.

### **2️⃣ LSTM – Remaining Useful Life (RUL)**
Predicts how many cycles are left before failure.

### **3️⃣ Fusion Classifier**
Final stage that combines:
- Anomaly score
- RUL

Output → **Failure Probability (0–1)**

---

## 🖥️ Streamlit Dashboard
Features:
- CSV Upload Page
- Real‑time Health Index
- Failure Probability indicator
- RUL estimation
- Sensor trend visualizations

Add screenshot here:
```
![Dashboard](assets/dashboard.png)
```

---

## ▶ How to Run
```bash
cd dashboard
streamlit run app.py
```
Upload your sensor CSV with:
```
vibration, temperature, pressure, torque, current, rpm
```

---

## 📦 Requirements
```
tensorflow==2.12.0
numpy
pandas
scikit-learn
streamlit
joblib
matplotlib
```
Install using:
```bash
pip install -r requirements.txt
```

---

## 🔍 Sample Output
```
Health Index: 0.34
Failure Probability: 1.00
Predicted RUL: 1675 steps
Machine Status: CRITICAL FAILURE
```

---

## 🌐 Where This Can Be Used
- CNC Machines
- Motors & Pumps
- HVAC Systems
- Turbines & Rotors
- Robotics Sensor Health
- Industrial IoT Systems

---

## 🚀 Future Enhancements
- GRU‑based RUL model
- Sensor Drift Compensation
- Multi‑Machine Monitoring
- Docker Deployment
- Real‑time Factory Alerts

---

## 👤 Author
**Vignesh KV**  
Final Year AI/ML Engineer — EWIT  
Passionate about ML Engineering & Industrial AI

🔗 LinkedIn: https://www.linkedin.com/in/vigneshkv200  
🐙 GitHub: https://github.com/vigneshkv200

---

## ⭐ Final Note
This project demonstrates **real ML Engineering** — combining Deep Learning, Hybrid Fusion, and a complete Streamlit deployment. A fully portfolio‑ready, recruiter‑friendly project.

