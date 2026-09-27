import pandas as pd


def load_adsb_data(file_path="data/adsb_data.csv"):
    """
    Load ADS-B observations from a CSV file.
    """

    df = pd.read_csv(file_path)

    return df