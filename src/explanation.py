import pandas as pd


def generate_explanation(row):

    reasons = []

    if row.get("speed_anomaly", False):
        reasons.append("unrealistic speed")

    if row.get("altitude_anomaly", False):
        reasons.append("abnormal altitude")

    if row.get("position_anomaly", False):
        reasons.append("sudden position change")

    if row.get("altitude_change_anomaly", False):
        reasons.append("large altitude change")

    if row.get("ml_anomaly_score", 0) >= 0.7:
        reasons.append("ML anomaly detected")

    if not reasons:
        return "Normal flight behavior"

    return " + ".join(reasons)


def add_explanations(df):

    df = df.copy()

    df["explanation"] = df.apply(
        generate_explanation,
        axis=1
    )

    return df