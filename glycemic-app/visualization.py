import matplotlib.pyplot as plt

def plot_glucose_curve(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["datetime"], df["glucose"], color="blue")
    ax.set_title("Courbe glycémique")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(feature_names, importances, color="steelblue")
    ax.set_title("Importance des variables")
    return fig
