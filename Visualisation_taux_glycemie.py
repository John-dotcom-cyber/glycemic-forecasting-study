import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv("glycemie_clean.csv", sep=',')
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date

# Sélection d'une journée
date_exemple = df['date'].iloc[0]
df_day = df[df['date'] == date_exemple].sort_values(by='datetime')

# Moyenne glissante sur 3 points
df_day['glycemie_smooth'] = df_day['3'].rolling(window=3, center=True).mean()

# Création de la figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- 1) Scatter + courbe lissée + seuils ---
axes[0].scatter(df_day['datetime'], df_day['3'], color='blue', s=20, label='Mesures')
axes[0].plot(df_day['datetime'], df_day['glycemie_smooth'], color='red', linewidth=2, label='Moyenne glissante')

# Seuils hypo/hyper
axes[0].axhline(70, color='green', linestyle='--', label='Hypoglycémie')
axes[0].axhline(180, color='orange', linestyle='--', label='Hyperglycémie')

axes[0].set_title(f"Glycémie dans le temps ({date_exemple})")
axes[0].set_xlabel("Temps")
axes[0].set_ylabel("Glycémie")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- 2) Histogramme lisible ---
df_day['3'] = pd.to_numeric(df_day['3'], errors='coerce')
df_day['glycemie_arrondie'] = df_day['3'].round()
axes[1].hist(df_day['glycemie_arrondie'], bins='auto', color='orange', edgecolor='black')
axes[1].set_title("Distribution des glycémies")
axes[1].set_xlabel("Glycémie")
axes[1].set_ylabel("Fréquence")
axes[1].tick_params(axis='x', labelrotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


