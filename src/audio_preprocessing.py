import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt
from .config import SAMPLE_RATE

def load_audio(path, sr=SAMPLE_RATE, mono=True):
    y, sr = librosa.load(path, sr=sr, mono=mono)
    if y is None or len(y) == 0:
        return np.zeros(1, dtype=np.float32), sr
    return y.astype(np.float32), sr

def normalize_audio(y, eps=1e-8):
    peak = np.max(np.abs(y)) if len(y) else 0
    if peak < eps:
        return y
    return y / peak

def bandpass_filter(y, sr, low=500, high=14000, order=4):
    nyq = sr / 2
    low = max(20, low) / nyq
    high = min(high, nyq - 100) / nyq
    if not 0 < low < high < 1:
        return y
    sos = butter(order, [low, high], btype="band", output="sos")
    try:
        return sosfiltfilt(sos, y).astype(np.float32)
    except Exception:
        return y

def preprocess_audio(y, sr, use_preemphasis=True, use_bandpass=True):
    y = normalize_audio(y)
    if use_bandpass:
        y = bandpass_filter(y, sr)
    if use_preemphasis:
        y = librosa.effects.preemphasis(y).astype(np.float32)
    return normalize_audio(y)

def augment_audio(y, sr, mode=None, random_state=42):
    rng = np.random.default_rng(random_state)
    if mode is None or mode == "none":
        return y
    if mode == "noise":
        return normalize_audio(y + rng.normal(0, 0.005, len(y)).astype(np.float32))
    if mode == "gain":
        return normalize_audio(y * float(rng.uniform(0.7, 1.3)))
    if mode == "shift":
        shift = int(rng.integers(-sr, sr))
        return np.roll(y, shift)
    return y

def audio_segments(y, sr, segment_seconds=5.0, overlap_seconds=2.5):
    seg_len = int(segment_seconds * sr)
    hop = max(1, int((segment_seconds - overlap_seconds) * sr))
    if len(y) <= seg_len:
        padded = np.pad(y, (0, max(0, seg_len - len(y))))
        yield 0, 0.0, segment_seconds, padded
        return
    seg_id = 0
    for start in range(0, len(y) - seg_len + 1, hop):
        end = start + seg_len
        yield seg_id, start / sr, end / sr, y[start:end]
        seg_id += 1
