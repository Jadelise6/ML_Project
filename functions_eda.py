# Fonctions importées dans l'EDA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import skew

# Style sns pour augmenter la taille des bordures, etc
sns.set_style("ticks")
sns.set_context("talk")

def missing_values_graph(df,df_str):
    """ Affiche le graphe des valeurs manquantes
    Args:
        - df (pd.DataFrame) : données considérées
        - df_str (str) : nom des données utilisées pour l'affichage et l'enregistrement
    """
    # Missing values graph (excluding columns with no missing values)
    plt.figure(figsize=(10, 7))

    missing_counts = df.isnull().sum().reset_index()
    missing_counts.columns = ["column", "missing_count"]

    # We remove rows that have a missing count of zero because they are of no information
    missing_counts = missing_counts[missing_counts["missing_count"] > 0]
    missing_counts = missing_counts.sort_values(by="missing_count", ascending=False)

    sns.set_style("ticks")
    sns.set_context("talk")
    ax = sns.barplot(data=missing_counts, x="missing_count", y="column", orient="h", color="dodgerblue")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(f"{df_str} - Nombre de valeurs manquantes")
    plt.ylabel("Variable")

    plt.tight_layout()
    plt.savefig(f"./Images/{df_str}_missing_values.png", format='png', bbox_inches='tight')
    plt.show()
    
def show_skewness(df, df_str, col, xlabel, max_value=None):
    """ Affiche le graphe des valeurs manquantes
    Args:
        - df (pd.DataFrame) : données considérées
        - df_str (str) : nom des données utilisées pour l'affichage et l'enregistrement
        - col (str) : nom de la colonne/variable considérée
        - xlabel (str) : nom de la colonne écrit pour le label de l'axe des abscisses
        - max_value (float) : valeur maximum où couper l'axe des x si nécessaire
    """
    if max_value:
        df = df[df <= max_value]
      
    # Calculs
    mean = round(np.mean(df),2)
    skewness = round(skew(df),2)
    median = round(np.median(df),2)
        
    # Affichage du graphe
    sns.set_context(rc = {'patch.linewidth': 0.0}) # pas d'espace entre les barres
    sns.histplot(data=df, color="dodgerblue", bins=60)
    plt.axvline(mean, color='red', label=f"Mean = {mean}")
    plt.axvline(median, color='blue', linestyle="--", label=f"Median = {median}")
    plt.title(f"{df_str} - {col} - Skewness : {skewness}")
    plt.xlabel(xlabel)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"./Images/{df_str}_{col}_skewness.png", format='png', bbox_inches='tight')
    plt.show()
    
ordinal_col = ["Growth_Stage", "Growth_Rate", "Difficulty_Level", "Fragrance_Level", "Winter_Care_Level", "Disease_Resistance_Level"]
