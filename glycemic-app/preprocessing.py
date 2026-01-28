import pandas as pd

def preprocess_data(df):
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime")
    return df
