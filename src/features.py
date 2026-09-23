import pandas as pd
import numpy as np


def create_features(df):

    df = df.copy()

    # Make sure data is ordered correctly
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        ["aircraft_id", "timestamp"]
    ).reset_index(drop=True)

    # Changes between consecutive observations
    df["latitude_change"] = (
        df.groupby("aircraft_id")["latitude"].diff().fillna(0)
    )

    df["longitude_change"] = (
        df.groupby("aircraft_id")["longitude"].diff().fillna(0)
    )

    df["altitude_change"] = (
        df.groupby("aircraft_id")["altitude"].diff().fillna(0)
    )

    df["speed_change"] = (
        df.groupby("aircraft_id")["speed"].diff().fillna(0)
    )

    # Position movement
    df["position_change"] = np.sqrt(
        df["latitude_change"] ** 2
        + df["longitude_change"] ** 2
    )

    # Absolute changes
    df["abs_altitude_change"] = df["altitude_change"].abs()

    df["abs_speed_change"] = df["speed_change"].abs()

    return df


if __name__ == "__main__":

    input_file = "data/adsb_data.csv"

    df = pd.read_csv(input_file)

    df = create_features(df)

    print("Features created successfully!")

    print(
        df[
            [
                "aircraft_id",
                "latitude",
                "longitude",
                "altitude",
                "speed",
                "position_change",
                "abs_altitude_change",
                "abs_speed_change"
            ]
        ].head(10)
    )