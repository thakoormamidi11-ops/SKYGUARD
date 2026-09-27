# SKYGUARD

## Explainable AI for ADS-B Airspace Anomaly Detection

SKYGUARD is a software-based prototype that detects suspicious aircraft behavior from ADS-B data and explains the evidence behind each alert.

The system analyzes aircraft position, speed, altitude, and movement changes using behavioral rules and anomaly scoring. The detected signals are combined into a risk score and presented through an interactive Streamlit dashboard.

---

## Problem

ADS-B (Automatic Dependent Surveillance-Broadcast) allows aircraft to broadcast information such as:

- Aircraft identity
- Latitude
- Longitude
- Altitude
- Speed
- Timestamp

ADS-B messages are not cryptographically authenticated. As a result, monitoring systems may receive false, inconsistent, or manipulated surveillance information.

Suspicious aircraft behavior can include:

- Sudden position changes
- Unrealistic speed
- Abnormal altitude
- Large altitude changes
- Unusual flight behavior

SKYGUARD focuses on detecting these suspicious patterns and helping an analyst understand why an observation was flagged.

---

## Solution

SKYGUARD uses a multi-signal approach combining:

- Behavioral rule checks
- Aircraft movement features
- Anomaly scoring
- Risk scoring
- Explainable alerts
- Geospatial visualization

### Detection Pipeline

```text
ADS-B Data
     ↓
Data Loading
     ↓
Feature Engineering
     ↓
Behavioral Rule Checks
     ↓
Anomaly Scoring
     ↓
Evidence Fusion
     ↓
Risk Score
     ↓
Explanation
     ↓
Streamlit Dashboard