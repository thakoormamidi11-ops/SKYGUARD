import pandas as pd
import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def calculate_ml_anomaly_score(df):
    df = df.copy()

    features = [
        "position_change",
        "abs_altitude_change",
        "abs_speed_change",
        "speed",
        "altitude"
    ]

    X = df[features].fillna(0)

    # =========================================================
    # ISOLATION FOREST
    # =========================================================

    if SKLEARN_AVAILABLE:

        model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )

        model.fit(X)

        predictions = model.predict(X)

        # Higher value = more anomalous
        raw_scores = -model.decision_function(X)

        min_score = raw_scores.min()
        max_score = raw_scores.max()

        if max_score > min_score:
            anomaly_scores = (
                (raw_scores - min_score)
                / (max_score - min_score)
            )
        else:
            anomaly_scores = pd.Series(
                0.0,
                index=df.index
            )

        df["ml_prediction"] = predictions
        df["ml_anomaly_score"] = anomaly_scores.round(3)

        return df

    # =========================================================
    # FALLBACK ANOMALY DETECTOR
    # =========================================================
    #
    # Used when Isolation Forest is unavailable.
    #
    # This is NOT machine learning.
    # It calculates severity from aircraft behavior.
    # =========================================================

    # ---------------------------------------------------------
    # 1. Position severity
    # ---------------------------------------------------------

    position_score = (
        1 - np.exp(
            -df["position_change"] / 0.5
        )
    )

    # ---------------------------------------------------------
    # 2. Altitude-change severity
    # ---------------------------------------------------------

    altitude_change_score = (
        1 - np.exp(
            -df["abs_altitude_change"] / 2000
        )
    )

    # ---------------------------------------------------------
    # 3. Speed-change severity
    # ---------------------------------------------------------

    speed_change_score = (
        1 - np.exp(
            -df["abs_speed_change"] / 100
        )
    )

    # ---------------------------------------------------------
    # 4. Absolute speed severity
    # ---------------------------------------------------------

    excessive_speed = (
        df["speed"] - 500
    ).clip(lower=0)

    speed_score = (
        1 - np.exp(
            -excessive_speed / 300
        )
    )

    # ---------------------------------------------------------
    # 5. Absolute altitude severity
    # ---------------------------------------------------------

    excessive_altitude = (
        df["altitude"] - 45000
    ).clip(lower=0)

    altitude_score = (
        1 - np.exp(
            -excessive_altitude / 10000
        )
    )

    # Very low altitude also contributes to severity
    low_altitude = (
        20000 - df["altitude"]
    ).clip(lower=0)

    low_altitude_score = (
        1 - np.exp(
            -low_altitude / 5000
        )
    )

    altitude_score = np.maximum(
        altitude_score,
        low_altitude_score
    )

    # ---------------------------------------------------------
    # Combine the signals
    # ---------------------------------------------------------

    fallback_score = (
        position_score * 0.30
        + altitude_change_score * 0.20
        + speed_change_score * 0.15
        + speed_score * 0.20
        + altitude_score * 0.15
    )

    fallback_score = (
        fallback_score
        .clip(0, 1)
        .round(3)
    )

    df["ml_anomaly_score"] = fallback_score

    # ---------------------------------------------------------
    # Decide whether the observation is anomalous
    # ---------------------------------------------------------

    df["ml_prediction"] = 1

    df.loc[
        df["ml_anomaly_score"] >= 0.7,
        "ml_prediction"
    ] = -1

    return df