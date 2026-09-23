# ✈️ SKYGUARD

## Explainable AI for ADS-B Airspace Anomaly Detection

SKYGUARD is a software-based airspace monitoring prototype that analyzes ADS-B aircraft observations and detects suspicious flight behavior.

The system combines rule-based detection with Isolation Forest anomaly detection and generates an explainable risk score for each observation.

---

## 🎯 Problem

ADS-B provides aircraft information such as:

- Position
- Altitude
- Speed
- Aircraft identity
- Time information

Suspicious or inconsistent observations can make airspace monitoring difficult, especially when large amounts of aircraft data must be analyzed.

SKYGUARD automatically identifies unusual behavior and provides evidence explaining why an observation was flagged.

---

## 💡 Solution

SKYGUARD follows this pipeline:

ADS-B Data  
↓  
Feature Extraction  
↓  
Rule-Based Detection  
↓  
Isolation Forest  
↓  
Evidence Fusion  
↓  
Risk Score  
↓  
Explainable Alert  
↓  
Map Visualization

---

## 🔍 Detection Signals

SKYGUARD currently analyzes:

### Kinematic Signals

- Aircraft speed
- Altitude
- Speed change
- Altitude change

### Trajectory Signals

- Latitude change
- Longitude change
- Position change

### Anomaly Signals

- Unrealistic speed
- Abnormal altitude
- Sudden position change
- Large altitude change
- Isolation Forest anomaly score

---

## 🤖 Machine Learning

SKYGUARD uses the **Isolation Forest** algorithm for unsupervised anomaly detection.

The model analyzes aircraft behavior features and identifies observations that differ significantly from normal patterns.

The ML result is combined with rule-based evidence to produce a final risk score.

---

## 📊 Risk Levels

| Risk Level | Meaning |
|---|---|
| NORMAL | No significant suspicious behavior |
| LOW | Limited anomaly evidence |
| MEDIUM | Multiple anomaly signals |
| HIGH | Strong anomaly evidence |

---

## 💡 Explainable Alerts

Instead of only showing an anomaly score, SKYGUARD provides an explanation.

Example:

> HIGH RISK — unrealistic speed + sudden position change

This helps an operator understand why an observation was flagged.

---

## 🗺️ Dashboard

The Streamlit dashboard provides:

- Aircraft tracking map
- Total observations
- Number of aircraft
- Alert count
- Risk levels
- Aircraft investigation
- Aircraft trajectory
- Isolation Forest score
- Detection explanation
- Risk distribution
- Demo filtering controls

---

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Isolation Forest
- Streamlit
- Plotly

---

## 📁 Project Structure

```text
SKYGUARD/
│
├── data/
│   ├── generate_data.py
│   ├── adsb_data.csv
│   ├── detected_anomalies.csv
│   └── final_results.csv
│
├── src/
│   ├── anomaly_detection.py
│   ├── features.py
│   ├── anomaly_model.py
│   ├── explanation.py
│   └── pipeline.py
│
├── app.py
├── requirements.txt
└── README.md