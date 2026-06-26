from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed
from .audio_preprocessing import load_audio, preprocess_audio, augment_audio, audio_segments
from .config import SAMPLE_RATE, CACHE_DIR, SEGMENT_SECONDS, SEGMENT_OVERLAP, RANDOM_STATE

def _stats(prefix, arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = {}
    for i, row in enumerate(arr):
        out[f"{prefix}_{i}_mean"] = float(np.mean(row))
        out[f"{prefix}_{i}_std"] = float(np.std(row))
        out[f"{prefix}_{i}_min"] = float(np.min(row))
        out[f"{prefix}_{i}_max"] = float(np.max(row))
    return out

def extract_features_from_signal(y, sr=SAMPLE_RATE):
    if len(y) < 32:
        return {}
    feats = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmin=500, fmax=min(14000, sr//2))
    logmel = librosa.power_to_db(mel + 1e-9)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats.update(_stats("mfcc", mfcc))
    feats.update(_stats("dmfcc", delta))
    feats.update(_stats("logmel", logmel))
    feats.update(_stats("chroma", chroma))
    feats.update(_stats("contrast", contrast))
    one_dim = {
        "rms": librosa.feature.rms(y=y)[0],
        "zcr": librosa.feature.zero_crossing_rate(y)[0],
        "centroid": librosa.feature.spectral_centroid(y=y, sr=sr)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr)[0],
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr)[0],
        "flatness": librosa.feature.spectral_flatness(y=y)[0],
    }
    for name, val in one_dim.items():
        feats[f"{name}_mean"] = float(np.mean(val))
        feats[f"{name}_std"] = float(np.std(val))
        feats[f"{name}_p10"] = float(np.percentile(val, 10))
        feats[f"{name}_p90"] = float(np.percentile(val, 90))
    feats["duration"] = float(len(y) / sr)
    return feats

def _cache_key(path, suffix="base"):
    p = Path(path)
    raw = f"{p.resolve()}_{p.stat().st_mtime if p.exists() else 0}_{suffix}".encode()
    return hashlib.md5(raw).hexdigest()

def extract_file_features(path, label=None, use_cache=True, force=False, augment_modes=("none",), max_seconds=30):
    rows = []
    for aug_idx, mode in enumerate(augment_modes):
        key = _cache_key(path, suffix=f"{mode}_{max_seconds}")
        cache_path = CACHE_DIR / f"{key}.pkl"
        if use_cache and cache_path.exists() and not force:
            row = pd.read_pickle(cache_path)
            rows.append(row)
            continue
        try:
            y, sr = load_audio(path, sr=SAMPLE_RATE)
            if max_seconds and len(y) > max_seconds * sr:
                y = y[: int(max_seconds * sr)]
            y = preprocess_audio(y, sr)
            y = augment_audio(y, sr, mode=mode, random_state=RANDOM_STATE + aug_idx)
            feats = extract_features_from_signal(y, sr)
            feats.update({"filepath": str(path), "filename": Path(path).name, "label": label, "augmentation": mode})
            row = pd.Series(feats)
            if use_cache:
                row.to_pickle(cache_path)
            rows.append(row)
        except Exception as e:
            rows.append(pd.Series({"filepath": str(path), "filename": Path(path).name, "label": label, "augmentation": mode, "error": str(e)}))
    return rows

def build_feature_table(df_files, label_col="label", n_jobs=-1, use_cache=True, force=False, augment=False, max_seconds=30):
    modes = ("none", "noise", "gain") if augment else ("none",)
    tasks = []
    for _, row in df_files.iterrows():
        tasks.append((row["filepath"], row.get(label_col, None)))
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(extract_file_features)(path, label, use_cache, force, modes, max_seconds) for path, label in tasks
    )
    flat = [r for sub in results for r in sub]
    return pd.DataFrame(flat)

def extract_soundscape_segments(path, segment_seconds=SEGMENT_SECONDS, overlap_seconds=SEGMENT_OVERLAP):
    y, sr = load_audio(path, sr=SAMPLE_RATE)
    y = preprocess_audio(y, sr)
    rows = []
    for seg_id, start, end, seg in audio_segments(y, sr, segment_seconds, overlap_seconds):
        feats = extract_features_from_signal(seg, sr)
        feats.update({"filename": Path(path).name, "filepath": str(path), "segment_id": seg_id, "start_sec": start, "end_sec": end})
        rows.append(feats)
    return pd.DataFrame(rows)
