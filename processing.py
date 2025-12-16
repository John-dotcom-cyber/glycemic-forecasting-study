# processing.py

import os
import pandas as pd

def load_and_merge_diabetes_data(path):
    files = [
        f for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.startswith("data-")
    ]

    if not files:
        raise ValueError("Aucun fichier data-* trouvé dans le dossier.")

    dfs = []
    for f in files:
        full_path = os.path.join(path, f)
        df = pd.read_csv(full_path, delim_whitespace=True, header=None)
        df["source"] = f
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
