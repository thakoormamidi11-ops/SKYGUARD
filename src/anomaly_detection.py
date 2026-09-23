import pandas as pd
import numpy as np

INPUT_FILE = "data/adsb_data.csv"
OUTPUT_FILE = "data/detected_anomalies.csv"


# Load ADS-B data
df = pd.read_csv(INPUT_FILE)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort aircraft observations by time
df = df.sort_values(["aircraft_id", "timestamp"]).reset_index(drop=True)


# Calculate changes between consecutive observations
df["latitude_change"] = df.groupby("aircraft_id")["latitude"].diff()
df["longitude_change"] = df.groupby("aircraft_id")["longitude"].diff()
df["altitude_change"] = df.groupby("aircraft_id")["altitude"].diff()
df["speed_change"] = df.groupby("aircraft_id")["speed"].diff()


# Distance-like movement measure
df["position_change"] = np.sqrt(
    df["latitude_change"].fillna(0) ** 2 +
    df["longitude_change"].fillna(0) ** 2
)


# -----------------------------
# Anomaly rules
# -----------------------------

df["speed_anomaly"] = df["speed"] > 800

df["altitude_anomaly"] = (
    (df["altitude"] < 20000) |
    (df["altitude"] > 45000)
)

df["position_anomaly"] = df["position_change"] > 0.5

df["altitude_change_anomaly"] = (
    df["altitude_change"].abs() > 2000
)


# -----------------------------
# Evidence count
# -----------------------------

df["evidence_count"] = (
    df["speed_anomaly"].astype(int)
    + df["altitude_anomaly"].astype(int)
    + df["position_anomaly"].astype(int)
    + df["altitude_change_anomaly"].astype(int)
)


# -----------------------------
# Risk score
# -----------------------------

df["risk_score"] = (
    df["evidence_count"] * 25
).clip(upper=100)


# Risk level
df["risk_level"] = np.select(
    [
        df["risk_score"] >= 75,
        df["risk_score"] >= 50,
        df["risk_score"] >= 25
    ],
    [
        "HIGH",
        "MEDIUM",
        "LOW"
    ],
    default="NORMAL"
)


# -----------------------------
# Explanation
# -----------------------------

def create_explanation(row):

    reasons = []

    if row["speed_anomaly"]:
        reasons.append("unrealistic speed")

    if row["altitude_anomaly"]:
        reasons.append("abnormal altitude")

    if row["position_anomaly"]:
        reasons.append("sudden position change")

    if row["altitude_change_anomaly"]:
        reasons.append("large altitude change")

    if not reasons:
        return "Normal flight behavior"

    return " + ".join(reasons)


df["explanation"] = df.apply(create_explanation, axis=1)


# Save results
df.to_csv(OUTPUT_FILE, index=False)

print("Anomaly detection completed!")
print(f"Results saved to: {OUTPUT_FILE}")

print("\nDetected anomalies:")
print(
    df[df["risk_level"] != "NORMAL"][
        [
            "aircraft_id",
            "timestamp",
            "risk_score",
            "risk_level",
            "explanation"
        ]
    ].to_string(index=False)
)