import streamlit as st
import pandas as pd
import pickle
import os
from preprocessing import preprocess_data
from features import compute_features
from visualization import plot_glucose_curve, plot_feature_importance

st.set_page_config(page_title="Glycemic Severity Predictor", layout="wide")

st.title("🔬 Glycemic Severity Predictor")
st.write("Prototype interactif pour analyser un profil glycémique et prédire un risque sévère.")

# section de téléchargement
st.markdown(""" ### 📁 Fichiers CSV d'exemple 
            Vous pouvez télécharger des fichiers de test directement depuis GitHub : 
            glycemic-app/patients_demo
            - [data-01-normal.csv](https://github.com/John-dotcom-cyber/glycemic-forecasting-study/tree/main/glycemic-app/patients_demo/data-01-normal.csv) 
            - [data-02-severe.csv](https://github.com/John-dotcom-cyber/glycemic-forecasting-study/tree/main/glycemic-app/patients_demo/data-02-severe.csv)
            - [data-03-instable.csv](https://github.com/John-dotcom-cyber/glycemic-forecasting-study/tree/main/glycemic-app/patients_demo/data-03-instable.csv) 
            - [data-04-modere.csv](https://github.com/John-dotcom-cyber/glycemic-forecasting-study/tree/main/glycemic-app/patients_demo/data-04-modere.csv)
            - [data-05-hypoglycemique.csv](https://github.com/John-dotcom-cyber/glycemic-forecasting-study/tree/main/glycemic-app/patients_demo/data-05-hypoglycemique.csv) 
""")
st.markdown("""
Ces fichiers peuvent être importés dans l'application via le bouton d'upload ci-dessus. """)

uploaded_file = st.file_uploader("📁 Importer un fichier CSV de mesures glycémiques", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Aperçu des données brutes")
    st.dataframe(df.head())

    # Prétraitement
    df_clean = preprocess_data(df)

    st.subheader("Courbe glycémique")
    st.pyplot(plot_glucose_curve(df_clean))

    # Features
    features = compute_features(df_clean)
    st.subheader("Variables dérivées")
    st.write(features)

    # Charger le modèle
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "Random_Forest.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Prédiction
    pred = model.predict(features.values.reshape(1, -1))[0]
    proba = model.predict_proba(features.values.reshape(1, -1))[0][1]

    st.subheader("🔍 Prédiction du modèle")
    if pred == 1:
        st.error(f"⚠️ Risque sévère détecté — probabilité : {proba:.2f}")
    else:
        st.success(f"🟢 Profil non sévère — probabilité : {proba:.2f}")

    # Importance des variables
    st.subheader("📊 Importance des variables")
    st.pyplot(plot_feature_importance(model, features.index))

    


