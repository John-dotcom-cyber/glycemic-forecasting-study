# Programme qui permet de convertir la date + heure en date

import pandas as pd

df = pd.read_csv("diabetes_merged.csv",sep=',')

# Créer une colonne datetime en ignorant les erreurs
df["datetime"] = pd.to_datetime(df["0"] + " " + df["1"], 
                                format="%m-%d-%Y %H:%M", 
                                errors="coerce")

# Supprimer les lignes où la conversion a échoué
df = df.dropna(subset=["datetime"])

# Vérification rapide
print(df.head())
print("Nombre de lignes restantes :", len(df))

# Sauvegarde dans un nouveau fichier CSV
df.to_csv("glycemie_clean.csv", index=False)

# Extraire la date seule
df["date"] = df["datetime"].dt.date

# Calculer les stats par jour
daily_stats = df.groupby("date")["3"].agg(["mean", "min", "max", "count"]).reset_index()

# Renommer les colonnes pour plus de clarté
daily_stats.columns = ["Date", "Glycemie_Moyenne", "Glycemie_Min", "Glycemie_Max", "Nb_Mesures"]

# Afficher les 10 premiers jours
print(daily_stats.head(10))
