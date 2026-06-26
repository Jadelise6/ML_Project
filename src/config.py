from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
CACHE_DIR = FEATURE_DIR / "cache"
MODEL_DIR = OUTPUT_DIR / "models"
BASELINE_MODEL_DIR = MODEL_DIR / "baselines"
FINAL_MODEL_DIR = MODEL_DIR / "final"
RESULT_DIR = OUTPUT_DIR / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"

TRAIN_AUDIO_DIR = DATA_DIR / "train_audio"
TRAIN_SOUNDSCAPES_DIR = DATA_DIR / "train_soundscapes"
TEST_SOUNDSCAPES_DIR = DATA_DIR / "test_soundscapes"
TRAIN_CSV = DATA_DIR / "train.csv"
SOUNDSCAPE_LABELS_CSV = DATA_DIR / "train_soundscapes_labels.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
TAXONOMY_CSV = DATA_DIR / "taxonomy.csv"

RANDOM_STATE = 42
SAMPLE_RATE = 32000
N_JOBS = 8
SEGMENT_SECONDS = 5.0
SEGMENT_OVERLAP = 2.5
TOP_K = 5
MIN_SAMPLES_PER_CLASS = 2

for d in [OUTPUT_DIR, FEATURE_DIR, CACHE_DIR, MODEL_DIR, BASELINE_MODEL_DIR, FINAL_MODEL_DIR, RESULT_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
