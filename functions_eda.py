# Fonctions importées dans l'EDA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import skew
import librosa

# Style sns pour augmenter la taille des bordures, etc
sns.set_style("ticks")
sns.set_context("talk")

def group_by_barplot(df, col_str, count_str):
    """ Affiche un bar plot utilisant value_counts
    Args:
        - df (pd.DataFrame) : données considérées
        - col_str (str) : colonne considérée
        - count_str (str) : colonne utilisée pour le count
    """
    plt.figure(figsize=(10, 7))

    ax = sns.barplot(data=df.value_counts(), orient="h", color="dodgerblue")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.ylabel(f"{col_str}")
    plt.xlabel(f"Nombre {count_str}")
    plt.title(f"Nombre {count_str} pour chaque {col_str}")
    plt.tight_layout()
    plt.savefig(f"./Images/{col_str}_valuecount.png", format='png', bbox_inches='tight')
    plt.show()
    
def show_skewness(df, df_str, col, xlabel, max_value=None):
    """ Affiche la skewness des données
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


def waveshow(path, data_str):
    """ Trace l'onde acoustique des données
        - path (str) : chemin du fichier audio
        - data_str (str) : titre pour l'affichage
    """
    y, sr = librosa.load(path, sr=None)
    
    plt.figure(figsize=(12, 5))
    librosa.display.waveshow(y=y, sr=sr)
    plt.xlabel("Temps")
    plt.title(f'Onde acoustique {data_str}')
    
    plt.tight_layout()
    plt.savefig(f"./Images/{data_str}_waveshow.png", format='png', bbox_inches='tight')
    plt.show()
    
def specshow(path, data_str):
    """ Trace le spectogramme des données
        - path (str) : chemin du fichier audio
        - data_str (str) : titre pour l'affichage
    """
    y, sr = librosa.load(path, sr=None)
    
    plt.figure(figsize=(10, 5))
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000) 
    spectrogram = librosa.power_to_db(spectrogram)
    librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time')
    plt.title(f'Spectogramme {data_str}')
    plt.xlabel("Temps")
    plt.ylabel("Fréquence (Hz)")
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(f"./Images/{data_str}_spectshow.png", format='png', bbox_inches='tight')
    plt.show()
    

def specshow_mfcc(path, data_str):
    """ Trace le spectogramme des MFCC des données
        - path (str) : chemin du fichier audio
        - data_str (str) : titre pour l'affichage
    """
    y, sr = librosa.load(path, sr=None)
    
    plt.figure(figsize=(14, 4))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    librosa.display.specshow(mfcc, x_axis='time')
    plt.title(f'Spectogramme MFCC {data_str}')
    plt.xlabel("Temps")
    plt.ylabel("Fréquence (Hz)")
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(f"./Images/{data_str}_spectshow_mfcc.png", format='png', bbox_inches='tight')
    plt.show()
    
def compare_mfcc(df, col, col_str, family, family_special=False):
    """ Compare les MFCC des données de dix espèces
        - df (str) : données
        - col (str) : colonne
        - col_str (str) : titre pour l'affichage
        - family (str) : famille considérée
    """
    # Prendre 10 espèces/familles aléatoirement
    np.random.seed(42)
    df_values = df[col].dropna().unique()
    size_sample = min(10, len(df_values))
    if size_sample == 0:
        return
    df_col_sample = np.random.choice(df_values, size_sample, replace=False)

    # Sélectionner les lignes correspondant à chaque espèce/famille
    rows = df[df[col].isin(df_col_sample)].copy()

    # Plot mean +- std par espèce
    x = np.arange(13)
    plt.figure(figsize=(12, 6))

    # Plot pour chaque espèce
    for species, row in rows.groupby(col):
        mat = np.vstack(row["MFCC"].apply(lambda v: np.fromstring(v.strip("[]"), sep=" ")).to_numpy())
        mu = mat.mean(axis=0)
        sd = mat.std(axis=0)

        x = np.arange(len(mu))
        plt.plot(x, mu, label=str(species), linewidth=2)
        plt.fill_between(x, mu - sd, mu + sd, alpha=0.15)

    plt.xticks(x, [f"{i}" for i in x])
    plt.xlabel("Coefficient MFCC")
    plt.ylabel("Valeur")
    plt.title(f"{family} - Moyenne MFCC par {col_str} avec variance (+-1 std)")
    plt.legend(ncol=2, fontsize=12)
    
    plt.tight_layout()
    if col == "common_name":
        plt.savefig(f"./Images/{col}_{family}_MFCC.png", format='png', bbox_inches='tight')
    else:
        plt.savefig(f"./Images/{col}_MFCC.png", format='png', bbox_inches='tight')
    plt.show()