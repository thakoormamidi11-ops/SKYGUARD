import pandas as pd

from features import create_features
from anomaly_model import calculate_ml_anomaly_score
from explanation import add_explanations


INPUT_FILE = "data/adsb_data.csv"
OUTPUT_FILE = "data/final_results.csv"


def run_pipeline():

    # 1. Load ADS-B data
    df = pd.read_csv(INPUT_FILE)

    # 2. Create movement/trajectory features
    df = create_features(df)

    # 3. Rule-based anomaly detection
    df["speed_anomaly"] = df["speed"] > 800

    df["altitude_anomaly"] = (
        (df["altitude"] < 20000)
        | (df["altitude"] > 45000)
    )

    df["position_anomaly"] = (
        df["position_change"] > 0.5
    )

    df["altitude_change_anomaly"] = (
        df["abs_altitude_change"] > 2000
    )

    # 4. Isolation Forest
    df = calculate_ml_anomaly_score(df)

    # 5. Count rule-based evidence
    df["evidence_count"] = (
        df["speed_anomaly"].astype(int)
        + df["altitude_anomaly"].astype(int)
        + df["position_anomaly"].astype(int)
        + df["altitude_change_anomaly"].astype(int)
    )

    # 6. Combine rules + ML score
    df["risk_score"] = (
        df["evidence_count"] * 20
        + df["ml_anomaly_score"] * 40
    ).clip(upper=100).round(1)

    # 7. Risk level
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

    # 8. Generate explanation
    df = add_explanations(df)

    # 9. Save final results
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return df


if __name__ == "__main__":

    results = run_pipeline()

    print("SKYGUARD pipeline completed!")

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("\nAlerts:")

    alerts = results[
        results["risk_level"] != "NORMAL"
    ]

    print(
        alerts[
            [
                "aircraft_id",
                "timestamp",
                "risk_score",
                "risk_level",
                "explanation"
            ]
        ].to_string(index=False)
    )