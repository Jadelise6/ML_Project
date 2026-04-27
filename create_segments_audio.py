import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# Chargement des données
train_audio_csv = Path("eda_csv/train_audio_features.csv")
train_audio_file = pd.read_csv(train_audio_csv)
csv_base_dir = train_audio_csv.parent
data_root = (csv_base_dir.parent / "data").resolve()
segments_dir = Path("segments")
segments_dir.mkdir(parents=True, exist_ok=True)

# Nettoyage des fichiers audios
def extract_active_segments(path):
    """ Suppression du silence/bruits de fond de faible intensité en découpant l'audio en segments actifs, c'est à dire
    en ne gardant que les segments qui ont un niveau sonore au plus 25 db sous le pic de l'audio
    Args:
        - path (str) : chemin du fichier audio
    """
    audio_path = (csv_base_dir / Path(path)).resolve()
    if not audio_path.exists():
        print(f"Fichier introuvable, ignoré: {path}")
        return

    try:
        audio, sr = librosa.load(audio_path, sr=None)
    except Exception as error:
        print(f"Impossible de lire, ignoré: {path} ({error})")
        return

    intervals = librosa.effects.split(audio, top_db=25) # 25db sous le pic acceptés
    for i, (start, end) in enumerate(intervals):
        chunk = np.ascontiguousarray(audio[start:end], dtype=np.float32)
        
        # On ignore les segments trop courts
        if len(chunk) < (sr * 0.5):
            continue
        
        # Création d'un nouveau fichier pour le segment
        relative_audio_path = audio_path.relative_to(data_root)
        output_dir = segments_dir / relative_audio_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        new_path = output_dir / f"{relative_audio_path.stem}_seg_{i}.wav"
        
        sf.write(new_path, chunk, sr, format="WAV", subtype="PCM_16")

def main():
    print("Début")

    cpt = 0
    for path in train_audio_file["filepath"]:
        extract_active_segments(path)
        cpt += 1

        if cpt % 50 == 0:
            print(cpt)

    print("Terminé !")


if __name__ == "__main__":
    main()