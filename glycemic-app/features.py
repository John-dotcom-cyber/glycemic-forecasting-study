import numpy as np
import pandas as pd

def compute_features(df):
    glucose = df["glucose"]

    features = {
        "mean_glucose": glucose.mean(),
        "median_glucose": glucose.median(),
        "cv": glucose.std() / glucose.mean(),
        "amplitude": glucose.max() - glucose.min(),
        "pct_high": (glucose > 180).mean() * 100,
        "pct_low": (glucose < 70).mean() * 100,
    }

    return pd.Series(features)
