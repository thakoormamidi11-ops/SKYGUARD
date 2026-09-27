import pandas as pd

from src.data_loader import load_adsb_data
from src.features import create_features
from src.anomaly_model import calculate_ml_anomaly_score
from src.explanation import add_explanations


INPUT_FILE = "data/adsb_data.csv"
OUTPUT_FILE = "data/final_results.csv"


def run_pipeline():

    # =========================================================
    # 1. LOAD ADS-B DATA
    # =========================================================

    df = load_adsb_data(INPUT_FILE)

    # =========================================================
    # 2. CREATE FEATURES
    # =========================================================

    df = create_features(df)

    # =========================================================
    # 3. RULE-BASED ANOMALY DETECTION
    # =========================================================

    # Unrealistic speed
    df["speed_anomaly"] = (
        df["speed"] > 800
    )

    # Abnormal altitude
    df["altitude_anomaly"] = (
        (df["altitude"] < 20000)
        | (df["altitude"] > 45000)
    )

    # Sudden position change
    df["position_anomaly"] = (
        df["position_change"] > 0.5
    )

    # Large altitude change
    df["altitude_change_anomaly"] = (
        df["abs_altitude_change"] > 2000
    )

    # =========================================================
    # 4. ML / FALLBACK ANOMALY DETECTION
    # =========================================================

    df = calculate_ml_anomaly_score(df)

    # =========================================================
    # 5. COUNT EVIDENCE
    # =========================================================

    df["evidence_count"] = (
        df["speed_anomaly"].astype(int)
        + df["altitude_anomaly"].astype(int)
        + df["position_anomaly"].astype(int)
        + df["altitude_change_anomaly"].astype(int)
    )

    # =========================================================
    # 6. CALCULATE INDIVIDUAL RISK COMPONENTS
    # =========================================================

    # ---------------------------------------------------------
    # Speed risk
    # ---------------------------------------------------------

    speed_risk = (
        ((df["speed"] - 800) / 400) * 100
    ).clip(0, 100)

    # ---------------------------------------------------------
    # Altitude risk
    # ---------------------------------------------------------

    high_altitude_risk = (
        ((df["altitude"] - 45000) / 15000) * 100
    ).clip(0, 100)

    low_altitude_risk = (
        ((20000 - df["altitude"]) / 10000) * 100
    ).clip(0, 100)

    altitude_risk = pd.concat(
        [high_altitude_risk, low_altitude_risk],
        axis=1
    ).max(axis=1)

    # ---------------------------------------------------------
    # Position risk
    # ---------------------------------------------------------

    position_risk = (
        ((df["position_change"] - 0.5) / 5) * 100
    ).clip(0, 100)

    # ---------------------------------------------------------
    # Altitude-change risk
    # ---------------------------------------------------------

    altitude_change_risk = (
        ((df["abs_altitude_change"] - 2000) / 5000) * 100
    ).clip(0, 100)

    # ---------------------------------------------------------
    # ML risk
    # ---------------------------------------------------------

    ml_risk = (
        df["ml_anomaly_score"] * 100
    )

    # =========================================================
    # 7. COMBINE RISK COMPONENTS
    # =========================================================

    # Use the strongest concrete signal as the main risk
    # and allow ML to increase confidence.

    strongest_rule_risk = pd.concat(
        [
            speed_risk,
            altitude_risk,
            position_risk,
            altitude_change_risk
        ],
        axis=1
    ).max(axis=1)

    df["risk_score"] = (
        strongest_rule_risk * 0.70
        + ml_risk * 0.30
    )

    # Multiple independent signals increase confidence.
    df.loc[
        df["evidence_count"] >= 2,
        "risk_score"
    ] += 10

    # Cap the final score.
    df["risk_score"] = (
        df["risk_score"]
        .clip(0, 100)
        .round(1)
    )

    # =========================================================
    # 8. ASSIGN RISK LEVEL
    # =========================================================

    df["risk_level"] = "NORMAL"

    df.loc[
        df["risk_score"] >= 25,
        "risk_level"
    ] = "LOW"

    df.loc[
        df["risk_score"] >= 50,
        "risk_level"
    ] = "MEDIUM"

    df.loc[
        df["risk_score"] >= 75,
        "risk_level"
    ] = "HIGH"

    # =========================================================
    # 9. GENERATE EXPLANATION
    # =========================================================

    df = add_explanations(df)

    # =========================================================
    # 10. SAVE RESULTS
    # =========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return df