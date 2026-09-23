import pandas as pd
from sklearn.ensemble import IsolationForest


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

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    model.fit(X)

    # Isolation Forest:
    # -1 = anomaly
    #  1 = normal
    predictions = model.predict(X)

    # Convert model decision score into 0-1 anomaly score
    raw_scores = -model.decision_function(X)

    min_score = raw_scores.min()
    max_score = raw_scores.max()

    if max_score > min_score:
        anomaly_scores = (
            (raw_scores - min_score)
            / (max_score - min_score)
        )
    else:
        anomaly_scores = 0

    df["ml_prediction"] = predictions
    df["ml_anomaly_score"] = anomaly_scores.round(3)

    return df


if __name__ == "__main__":

    input_file = "data/adsb_data.csv"

    df = pd.read_csv(input_file)

    df["latitude_change"] = (
        df.groupby("aircraft_id")["latitude"]
        .diff()
        .fillna(0)
    )

    df["longitude_change"] = (
        df.groupby("aircraft_id")["longitude"]
        .diff()
        .fillna(0)
    )

    df["altitude_change"] = (
        df.groupby("aircraft_id")["altitude"]
        .diff()
        .fillna(0)
    )

    df["speed_change"] = (
        df.groupby("aircraft_id")["speed"]
        .diff()
        .fillna(0)
    )

    df["position_change"] = (
        df["latitude_change"] ** 2
        + df["longitude_change"] ** 2
    ) ** 0.5

    df["abs_altitude_change"] = (
        df["altitude_change"].abs()
    )

    df["abs_speed_change"] = (
        df["speed_change"].abs()
    )

    df = calculate_ml_anomaly_score(df)

    print("Isolation Forest completed!")

    print(
        df[
            [
                "aircraft_id",
                "speed",
                "altitude",
                "ml_prediction",
                "ml_anomaly_score"
            ]
        ].head(20).to_string(index=False)
    )