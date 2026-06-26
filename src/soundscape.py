import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from .features import extract_soundscape_segments
from .modeling import decision_scores, normalize_scores, top_k_labels
from .data_loading import parse_multilabel
from .config import TOP_K

def predict_one_soundscape(path, bundles, k=TOP_K):
    project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    cache_dir = project_root / "outputs" / "features" / "soundscape_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{Path(path).stem}.parquet"

    if cache_path.exists():
        seg_df = pd.read_parquet(cache_path)
    else:
        seg_df = extract_soundscape_segments(path)
        seg_df.to_parquet(cache_path, index=False)
    if seg_df.empty:
        return pd.DataFrame()
    
    all_scores = []
    encoder = bundles[0]["label_encoder"]
    feature_cols = bundles[0]["feature_cols"]
    X = seg_df.reindex(columns=feature_cols, fill_value=0)
    for bundle in bundles:
        model = bundle["model"]
        scores = normalize_scores(decision_scores(model, X))
        all_scores.append(scores)
    avg_scores = np.mean(all_scores, axis=0)
    labels, vals = top_k_labels(avg_scores, encoder, k=k)
    rows = []
    for i in range(len(seg_df)):
        rows.append({
            "filename": seg_df.iloc[i]["filename"],
            "segment_id": int(seg_df.iloc[i]["segment_id"]),
            "start_sec": float(seg_df.iloc[i]["start_sec"]),
            "end_sec": float(seg_df.iloc[i]["end_sec"]),
            "top_labels": ";".join(labels[i]),
            "top_scores": ";".join([f"{x:.6f}" for x in vals[i]]),
            "max_score": float(np.max(avg_scores[i])),
        })
    return pd.DataFrame(rows)

def predict_soundscapes(file_df, bundles, n_jobs=-1, k=TOP_K):
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(predict_one_soundscape)(row.filepath, bundles, k) for row in file_df.itertuples(index=False)
    )
    parts = [p for p in parts if p is not None and not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def labels_from_scores(top_labels, top_scores, threshold=0.35, min_labels=1):
    labels = parse_multilabel(top_labels)
    scores = [float(x) for x in str(top_scores).split(";") if x != ""]
    kept = [lab for lab, sc in zip(labels, scores) if sc >= threshold]
    if len(kept) < min_labels and labels:
        kept = labels[:min_labels]
    return kept

def tune_threshold(labels_df, pred_df, label_col=None, thresholds=None):
    from .evaluation import multilabel_summary
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    if label_col is None:
        label_col = next((c for c in ["birds", "labels", "primary_label"] if c in labels_df.columns), None)
    if label_col is None:
        raise ValueError("Aucune colonne label trouvée")
    n = min(len(labels_df), len(pred_df))
    true_lists = labels_df.iloc[:n][label_col].apply(parse_multilabel).tolist()
    mlb = MultiLabelBinarizer()
    Y_true = mlb.fit_transform(true_lists)
    rows = []
    for th in thresholds:
        pred_lists = [labels_from_scores(r.top_labels, r.top_scores, threshold=th) for r in pred_df.iloc[:n].itertuples(index=False)]
        pred_lists = [[x for x in xs if x in mlb.classes_] for xs in pred_lists]
        Y_pred = mlb.transform(pred_lists)
        row = {"threshold": float(th), **multilabel_summary(Y_true, Y_pred)}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
