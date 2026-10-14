"""Métricas de alerta y elección de umbral."""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score, roc_curve


def metrics_at(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "recall": tp / (tp + fn) if tp + fn else 0.0,          # ataques detectados
        "precision": tp / (tp + fp) if tp + fp else 0.0,       # alertas correctas
        "fpr": fp / (fp + tn) if fp + tn else 0.0,             # normales alarmadas
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "accuracy": (tp + tn) / len(y),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    """Independientes del umbral."""
    if len(np.unique(score)) == 1:
        return {"roc_auc": 0.5, "pr_auc": float(y.mean())}
    return {"roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score)}


def threshold_for_max_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float) -> float:
    """Umbral con mayor recall cuya tasa de falsos positivos no supera max_fpr."""
    fpr, tpr, thr = roc_curve(y, score)
    ok = np.flatnonzero(fpr <= max_fpr)
    best = ok[np.argmax(tpr[ok])]
    return float(min(thr[best], 1.0)) if np.isfinite(thr[best]) else 1.0 + 1e-9
