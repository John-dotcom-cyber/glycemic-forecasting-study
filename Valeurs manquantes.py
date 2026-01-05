import pandas as pd

df = pd.read_csv("glycemie_clean.csv",sep=',')

# Compter les valeurs manquantes par colonne
missing_counts = df.isna().sum()

# Proportion de valeurs manquantes par colonne
missing_ratio = df.isna().mean() * 100

print("Valeurs manquantes par colonne :")
print(missing_counts)

print("\nProportion (%) de valeurs manquantes par colonne :")
print(missing_ratio)

# Créer la colonne datetime
#df["datetime"] = pd.to_datetime(df["0"] + " " + df["1"], 
#                                format="%m-%d-%Y %H:%M", 
#                                errors="coerce")

# Vérifier
#print(df[["0","1","datetime"]].head())

# Créer la colonne datetime df["datetime"] = pd.to_datetime(df["0"] + " " + df["1"], format="%m-%d-%Y %H:%M", errors="coerce") # Vérifier print(df[["0","1","datetime"]].head())
#Vérifie si les valeurs manquantes sont réparties aléatoirement ou concentrées sur certaines périodes/jours.
#df["missing"] = df["3"].isna()
#print(df.groupby(df["datetime"].dt.date)["missing"].mean())

print(df.columns)
# Suppression des lignes manquantes car 11% de valeurs manquentes sur 29285
df = df.dropna(subset=["3"])

# Sauvegarde dans un nouveau fichier CSV
df.to_csv("glycemie_clean.csv", index=False)
