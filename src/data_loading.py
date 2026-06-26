from pathlib import Path
import pandas as pd
from .config import TRAIN_AUDIO_DIR, TRAIN_SOUNDSCAPES_DIR, TEST_SOUNDSCAPES_DIR, TRAIN_CSV

AUDIO_EXTENSIONS = {".ogg", ".wav", ".mp3", ".flac", ".m4a"}

def scan_audio_dir(audio_dir: Path, recursive: bool = True) -> pd.DataFrame:
    audio_dir = Path(audio_dir)
    rows = []
    if not audio_dir.exists():
        return pd.DataFrame(columns=["filepath", "filename", "stem", "label"])
    iterator = audio_dir.rglob("*") if recursive else audio_dir.glob("*")
    for path in iterator:
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            label = path.parent.name if path.parent != audio_dir else None
            rows.append({"filepath": str(path), "filename": path.name, "stem": path.stem, "label": label})
    return pd.DataFrame(rows)

def load_train_metadata() -> pd.DataFrame:
    if TRAIN_CSV.exists():
        return pd.read_csv(TRAIN_CSV)
    return pd.DataFrame()

def scan_train_audio() -> pd.DataFrame:
    files = scan_audio_dir(TRAIN_AUDIO_DIR, recursive=True)
    meta = load_train_metadata()
    if not meta.empty and "filename" in meta.columns:
        files = files.merge(meta, how="left", on="filename", suffixes=("", "_meta"))
        if "primary_label" in files.columns:
            files["label"] = files["primary_label"].fillna(files["label"])
    return files

def scan_soundscapes(train: bool = True) -> pd.DataFrame:
    return scan_audio_dir(TRAIN_SOUNDSCAPES_DIR if train else TEST_SOUNDSCAPES_DIR, recursive=False)

def parse_multilabel(value):
    if pd.isna(value):
        return []
    value = str(value).strip()
    if value.lower() in {"nocall", "none", "nan", ""}:
        return []
    return [x.strip() for x in value.replace(",", ";").split(";") if x.strip() and x.strip().lower() != "nocall"]
