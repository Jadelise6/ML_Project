import librosa
import numpy as np
import soundfile as sf
from collections import defaultdict
from pathlib import Path


SEGMENTS_DIR = Path("segments")
AUGMENTATION_DIR = Path("augmentation")
AUGMENTATION_DIR.mkdir(parents=True, exist_ok=True)


def _load_audio(path):
    return librosa.load(path, sr=None)


def _output_path(input_path, kind, cpt):
    relative_path = input_path.resolve().relative_to(SEGMENTS_DIR.resolve())
    output_dir = AUGMENTATION_DIR / relative_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{relative_path.stem}_{kind}_{cpt}.wav"


def _write_audio(path, audio, sr, kind, cpt):
    output_path = _output_path(path, kind, cpt)
    sf.write(output_path, np.ascontiguousarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")


def noise_addition(path, cpt):
    """ Ajoute du bruit en utilisant une distribution normale N(0, 1)
    Valeur du facteur de bruit admissible = x > 0.004
    Args:
        - path (str) : chemin du fichier audio
        - cpt (int) : comptage actuel
    """
    audio, sr = _load_audio(path)

    # Ajout de bruit gaussien.
    audio_noise = audio + 0.009 * np.random.normal(0, 1, len(audio))
    _write_audio(path, audio_noise, sr, "noise_add", cpt)

def shifting(path, cpt):
    """ Déplace le spectre
    Valeur du facteur admissible = sr/10
    Args:
        - path (str) : chemin du fichier audio
        - cpt (int) : comptage
    """
    audio, sr = _load_audio(path)

    # Ajout du décalage.
    audio_shift = np.roll(audio, int(sr / 10))
    _write_audio(path, audio_shift, sr, "shift", cpt)
    
def time_stretching(path, cpt):
    """ Étire le temps des audios pour les ralentir
    Valeur du facteur admissible = 0.8 < x < 1.2
    Args:
        - path (str) : chemin du fichier audio
        - cpt (int) : comptage
    """
    factor = 1.0
    audio, sr = _load_audio(path)

    # Ajout de l'etirement temporel.
    audio_t_sttch = librosa.effects.time_stretch(audio, rate=factor)
    _write_audio(path, audio_t_sttch, sr, "stretch", cpt)
    
def pitch_shifting(path, cpt, step):
    """ Déplace la hauteur du son
    Valeur du facteur admissible = -2 <= x <= 2
    Args:
        - path (str) : chemin du fichier audio
        - cpt (int) : comptage
    """
    audio, sr = _load_audio(path)

    # Ajout du décalage de hauteur.
    audio_pitch_sf = librosa.effects.pitch_shift(audio, sr=sr, n_steps=step)
    _write_audio(path, audio_pitch_sf, sr, f"pitch_{step:+d}", cpt)
    
def pitch_shifting_wrapper(path, cpt):
    # Permet d'appeler pitch_shifting avec step
    pitch_shifting(path, cpt, step=np.random.randint(-2, 3))


def _segment_files_by_species():
    grouped = defaultdict(list)
    for segment_path in sorted(SEGMENTS_DIR.rglob("*.wav")):
        if segment_path.is_file():
            grouped[segment_path.parent.name].append(segment_path)
    return grouped


def main():
    # Boucle sur chaque espèce et augmentation du nombre d'échantillons
    nb_augm = 10
    augm_functions = [noise_addition, shifting, time_stretching, pitch_shifting_wrapper]
    species_to_segments = _segment_files_by_species()

    for species_id, segment_paths in species_to_segments.items():
        print("Espèce traitée", species_id)

        if len(segment_paths) == 0:
            continue

        cpt_augm_per_species = 0
        restart = True

        while restart:
            for segment_path in segment_paths:
                if cpt_augm_per_species >= nb_augm:
                    restart = False
                    break

                funct = np.random.choice(augm_functions)
                funct(segment_path, cpt_augm_per_species)
                cpt_augm_per_species += 1


if __name__ == "__main__":
    main()