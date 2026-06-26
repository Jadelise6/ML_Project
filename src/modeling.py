import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from .config import RANDOM_STATE, MIN_SAMPLES_PER_CLASS
from sklearn.model_selection import StratifiedKFold
def prepare_xy(features, label_col="label", min_samples_per_class=MIN_SAMPLES_PER_CLASS):
    df = features.copy()
    df = df[df[label_col].notna()].copy()
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    counts = df[label_col].value_counts()
    keep = counts[counts >= min_samples_per_class].index
    df_main = df[df[label_col].isin(keep)].copy()
    df_rare = df[~df[label_col].isin(keep)].copy()
    drop_cols = {label_col, "filepath", "filename", "stem", "error", "augmentation", "primary_label", "secondary_labels", "type"}
    feature_cols = [c for c in df_main.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df_main[c])]
    X = df_main[feature_cols].replace([np.inf, -np.inf], np.nan)
    le = LabelEncoder()
    y = le.fit_transform(df_main[label_col])
    return X, y, le, feature_cols, df_main, df_rare

def make_pipeline(clf, scale=True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)

def build_models(preset="balanced"):
    models = {
        "dummy_most_frequent": make_pipeline(DummyClassifier(strategy="most_frequent"), scale=False),
        "cosine_knn": make_pipeline(KNeighborsClassifier(n_neighbors=7, metric="cosine", weights="distance", n_jobs=8), scale=True),
        "linear_svm_balanced": make_pipeline(SGDClassifier(loss="hinge",penalty="l2",alpha=1e-4,class_weight="balanced",max_iter=2000,tol=1e-3,n_jobs=8,random_state=RANDOM_STATE), scale=True),
        "sgd_logistic_fast": make_pipeline(SGDClassifier(loss="log_loss", penalty="l2",alpha=1e-4, class_weight="balanced", max_iter=2000, tol=1e-3, n_jobs=8, random_state=RANDOM_STATE), scale=True),
        "random_forest": make_pipeline(RandomForestClassifier(n_estimators=150, max_depth=22, min_samples_leaf=5,min_samples_split=10,max_features="sqrt", class_weight="balanced_subsample", n_jobs=8, random_state=RANDOM_STATE), scale=False),
        "extra_trees": make_pipeline(ExtraTreesClassifier(n_estimators=180, max_depth=22, min_samples_leaf=5,min_samples_split=10,max_features="sqrt", class_weight="balanced", n_jobs=8, random_state=RANDOM_STATE), scale=False),
        "hist_gradient_boosting": make_pipeline(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,max_leaf_nodes=31, l2_regularization=0.05,early_stopping=True,validation_fraction=0.1,n_iter_no_change=10, random_state=RANDOM_STATE), scale=False),
        "adaboost_light": make_pipeline(AdaBoostClassifier(n_estimators=120, learning_rate=0.5, random_state=RANDOM_STATE), scale=False),
    }
    if preset == "fast":
        return {k: models[k] for k in ["dummy_most_frequent", "cosine_knn", "linear_svm_balanced", "sgd_logistic_fast", "extra_trees", "hist_gradient_boosting"]}
    return models

def save_bundle(path, model, label_encoder, feature_cols, metadata=None):
    payload = {"model": model, "label_encoder": label_encoder, "feature_cols": list(feature_cols), "metadata": metadata or {}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)

def load_bundle(path):
    return joblib.load(path)

def decision_scores(model, X):
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        s = model.predict_proba(X)
    else:
        pred = model.predict(X)
        s = np.zeros((len(pred), len(np.unique(pred))))
    if isinstance(s, list):
        s = np.vstack([x[:, 1] for x in s]).T
    return np.asarray(s)

def normalize_scores(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    mn = np.nanmin(scores, axis=1, keepdims=True)
    mx = np.nanmax(scores, axis=1, keepdims=True)
    return (scores - mn) / (mx - mn + 1e-9)

def top_k_labels(scores, label_encoder, k=5):
    idx = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    labels = label_encoder.inverse_transform(idx.ravel()).reshape(idx.shape)
    vals = np.take_along_axis(scores, idx, axis=1)
    return labels, vals

def filter_rare_classes(df, label_col="label", min_samples=5):
    counts = df[label_col].value_counts()
    valid_classes = counts[counts >= min_samples].index

    df_main = df[df[label_col].isin(valid_classes)].copy()
    df_rare = df[~df[label_col].isin(valid_classes)].copy()

    return df_main, df_rare, counts


def get_safe_cv(y, max_splits=5, random_state=42):
    counts = pd.Series(y).value_counts()
    min_count = counts.min()

    n_splits = min(max_splits, int(min_count))
    n_splits = max(2, n_splits)

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )