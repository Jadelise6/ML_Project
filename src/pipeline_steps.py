import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score
from .config import FEATURE_DIR, RESULT_DIR, BASELINE_MODEL_DIR, FINAL_MODEL_DIR, RANDOM_STATE, N_JOBS
from .data_loading import scan_train_audio, scan_soundscapes
from .features import build_feature_table
from .modeling import prepare_xy, build_models, save_bundle, load_bundle,get_safe_cv
from .evaluation import classification_summary, report_dataframe, confusion_dataframe
from .soundscape import predict_soundscapes, tune_threshold

def step_extract_train_features(force_recompute=False, augment=False, n_jobs=N_JOBS):
    parquet_path = FEATURE_DIR / "train_audio_features.parquet"
    csv_path = FEATURE_DIR / "train_audio_features.csv"
    if parquet_path.exists() and not force_recompute:
        return pd.read_parquet(parquet_path)
    if csv_path.exists() and not force_recompute:
        return pd.read_csv(csv_path)
    df_files = scan_train_audio()
    features = build_feature_table(df_files, n_jobs=n_jobs, augment=augment, force=force_recompute)
    features.to_parquet(parquet_path, index=False)
    features.to_csv(csv_path, index=False)
    return features

def step_compare_models(features, preset="fast", n_splits=5):
    X, y, encoder, feature_cols, df_main, df_rare = prepare_xy(features)
    cv = get_safe_cv(y,max_splits=n_splits,random_state=RANDOM_STATE)
    scoring = {
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
        "weighted_f1": make_scorer(f1_score, average="weighted", zero_division=0),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }
    rows = []
    models = build_models(preset=preset)
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
        row = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)
    results = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    results.to_csv(RESULT_DIR / "model_comparison_cv.csv", index=False)
    meta = {"n_samples": len(y), "n_classes": len(encoder.classes_), "n_rare_classes_excluded": df_rare["label"].nunique() if not df_rare.empty else 0}
    pd.DataFrame([meta]).to_csv(RESULT_DIR / "training_data_summary.csv", index=False)
    return results, meta

def step_train_and_save_all_baselines(features, preset="fast"):
    X, y, encoder, feature_cols, df_main, df_rare = prepare_xy(features)
    rows = []
    for name, model in build_models(preset=preset).items():
        model.fit(X, y)
        path = BASELINE_MODEL_DIR / f"{name}.joblib"
        save_bundle(path, model, encoder, feature_cols, metadata={"model_name": name})
        rows.append({"model": name, "path": str(path)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(RESULT_DIR / "saved_baseline_models.csv", index=False)
    return manifest

def step_error_analysis(features, model_names, n_splits=5):
    X, y, encoder, feature_cols, df_main, df_rare = prepare_xy(features)
    cv = get_safe_cv(y,max_splits=n_splits,random_state=RANDOM_STATE)
    rows = []
    for name in model_names:
        model = build_models(preset="balanced")[name]
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        summary = classification_summary(y, y_pred)
        summary["model"] = name
        rows.append(summary)
        report_dataframe(y, y_pred, encoder.classes_).to_csv(RESULT_DIR / f"classification_report_{name}.csv")
        confusion_dataframe(y, y_pred, encoder.classes_).to_csv(RESULT_DIR / f"confusion_matrix_{name}.csv")
        err = df_main.copy()
        err["true_label"] = encoder.inverse_transform(y)
        err["pred_label"] = encoder.inverse_transform(y_pred)
        err["is_error"] = err["true_label"] != err["pred_label"]
        err[err["is_error"]].to_csv(RESULT_DIR / f"errors_{name}.csv", index=False)
    out = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    out.to_csv(RESULT_DIR / "top_model_error_analysis_summary.csv", index=False)
    return out

def step_train_final_top3(features, top_model_names):
    X, y, encoder, feature_cols, df_main, df_rare = prepare_xy(features)
    manifest = []
    models = build_models(preset="balanced")
    for name in top_model_names:
        model = models[name]
        model.fit(X, y)
        path = FINAL_MODEL_DIR / f"final_{name}.joblib"
        save_bundle(path, model, encoder, feature_cols, metadata={"final": True, "model_name": name})
        manifest.append({"model": name, "path": str(path)})
    out = pd.DataFrame(manifest)
    out.to_csv(RESULT_DIR / "final_top3_models.csv", index=False)
    return out

def load_final_bundles(manifest_path=None):
    manifest_path = manifest_path or (RESULT_DIR / "final_top3_models.csv")
    manifest = pd.read_csv(manifest_path)
    return [load_bundle(p) for p in manifest["path"]]
