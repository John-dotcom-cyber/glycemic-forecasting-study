# main.py

from processing import load_and_merge_diabetes_data
import os
import pandas as pd

DATA_DIR = "data/diabetes-data-UCI"

def main():
    df = load_and_merge_diabetes_data(DATA_DIR)
    print(df.head())

    # Exemple : sauvegarde en CSV
    df.to_csv("diabetes_merged.csv", index=False)
    print("Fusion terminée !")

if __name__ == "__main__":
    main()
