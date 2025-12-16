# processing.py

import os
import pandas as pd

def load_and_merge_diabetes_data(data_dir):
    """
    Charge et fusionne tous les fichiers patients du dataset Diabetes UCI.
    Retourne un DataFrame Pandas.
    """

    all_rows = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            patient_id = filename.replace(".txt", "")
            filepath = os.path.join(data_dir, filename)

            df = pd.read_csv(
                filepath,
                sep="\t",
                header=None,
                names=["date", "time", "code", "value"],
                engine="python"
            )

            df["patient_id"] = patient_id
            all_rows.append(df)

    merged = pd.concat(all_rows, ignore_index=True)
    return merged