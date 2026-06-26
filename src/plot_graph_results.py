import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Paramètres
sns.set_style("ticks")
# Augmente légèrement l'échelle de police globale pour titres et labels
sns.set_context("talk", font_scale=1.25)
    
global_path_top_3 = "../outputs/results/"

all_results_path = "model_comparison_cv.csv"

path_top_3 = ["classification_report_cosine_knn.csv", "classification_report_extra_trees.csv", "classification_report_hist_gradient_boosting.csv"]
models_top_3 = ["KNN cosinus", "Extra Trees", "High Gradient Boosting"]

def results_all_models():
    # Lecture des fichiers de métriques
    df = pd.read_csv(global_path_top_3 + all_results_path)
    metrics_df = pd.DataFrame({"model": df["model"], "weighted_f1_mean": df["weighted_f1_mean"], "balanced_accuracy_mean": df["balanced_accuracy_mean"], "macro_f1_mean": df["macro_f1_mean"]})
    models_new_names = ["High Gradient Boosting", "Extra Trees", "KNN cosinus", "Régression logistique SGD", "SVM linéaire", "Dummy Classifier"]
    metrics_df["model"] = models_new_names
    metrics_df = metrics_df.rename(columns={"weighted_f1_mean": "Weighted-F1", "balanced_accuracy_mean": "Balanced accuracy", "macro_f1_mean": "Macro-F1"})

    # Construction du graphe
    order = metrics_df["model"].tolist()
    plot_df = metrics_df.melt(id_vars="model", value_vars=["Macro-F1", "Balanced accuracy", "Weighted-F1"], var_name="metric", value_name="score")

    # Figure plus compacte
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", order=order, ax=ax, palette="deep")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Modèle", fontsize=16)
    ax.set_ylabel("Score", fontsize=16)
    ax.set_title("Métriques d'évaluation sur les modèles", fontsize=18, fontweight="bold")
    ax.legend(prop={"size": 11})
    ax.yaxis.grid(True)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    plt.savefig("../outputs/figures/all_models_metrics_results.png", format="png", bbox_inches="tight")
    plt.show()
    
def top_3_plot_graph():
    # Création des listes pour sotcker les métriques
    precision = list()
    recall = list()
    f1_score = list()    
    
    # Lecture des fichiers de métriques
    for path in path_top_3:
        df = pd.read_csv(global_path_top_3 + path, index_col=0)
        row = df.loc["weighted avg", ["precision", "recall", "f1-score"]]        
        precision.append(float(row["precision"]))
        recall.append(float(row["recall"]))
        f1_score.append(float(row["f1-score"]))

    metrics_df = pd.DataFrame({"model": models_top_3, "precision": precision, "recall": recall, "f1-score": f1_score})

    # Construction du graphe
    order = metrics_df["model"].tolist()
    plot_df = metrics_df.melt(id_vars="model", value_vars=["precision", "recall", "f1-score"], var_name="metric", value_name="score")

    # Figure plus compacte
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", order=order, ax=ax, palette="deep")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Modèle", fontsize=16)
    ax.set_ylabel("Score", fontsize=16)
    ax.set_title("Métriques d'évaluation sur les trois meilleurs modèles", fontsize=18, fontweight="bold")
    ax.legend(prop={"size": 11})
    ax.yaxis.grid(True)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    plt.savefig("../outputs/figures/top_3_metrics_results.png", format="png", bbox_inches="tight")
    plt.show()