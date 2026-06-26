import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix, precision_score, recall_score

def classification_summary(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
    }

def report_dataframe(y_true, y_pred, labels):
    rep = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T

def confusion_dataframe(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=labels, columns=labels)

def multilabel_summary(Y_true, Y_pred):
    return {
        "macro_f1": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "samples_f1": f1_score(Y_true, Y_pred, average="samples", zero_division=0),
        "macro_precision": precision_score(Y_true, Y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(Y_true, Y_pred, average="macro", zero_division=0),
    }
