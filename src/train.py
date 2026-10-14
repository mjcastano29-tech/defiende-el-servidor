"""Entrenamiento de un experimento y registro en MLflow."""
import time

import mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import ROOT, git_commit
from src.data import TARGET, feature_columns, load_split, row_hash, xy
from src.features import TRUSTED_TYPES, build_preprocessor
from src.metrics import metrics_at, ranking_metrics, threshold_for_max_fpr


def setup_mlflow(params: dict) -> str:
    cfg = params["mlflow"]
    (ROOT / "mlflow").mkdir(exist_ok=True)
    uri = cfg["tracking_uri"].replace("sqlite:///", f"sqlite:///{ROOT}/")
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        mlflow.create_experiment(cfg["experiment"], artifact_location=(ROOT / cfg["artifact_dir"]).as_uri())
    mlflow.set_experiment(cfg["experiment"])
    return uri


def build_model(spec: dict, seed: int):
    kind = spec["model"]
    if kind == "dummy":
        return DummyClassifier(strategy="prior")
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=spec.get("class_weight"))
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=seed, class_weight=spec.get("class_weight"),
                                              **spec.get("hgb_params", {}))
    raise ValueError(f"Tipo de modelo desconocido: {kind!r}")


def run_experiment(spec: dict, params: dict) -> dict:
    setup_mlflow(params)
    train, val = load_split(params, "train"), load_split(params, "val")
    drop = spec.get("drop_features", [])
    if spec.get("dedup"):
        train = train.drop_duplicates(subset=feature_columns(train) + [TARGET])
    X_tr, y_tr = xy(train, drop)
    X_va, y_va = xy(val, drop)

    pipe = Pipeline([
        ("prep", build_preprocessor(list(X_tr.columns), params["preprocessing"]["min_frequency"])),
        ("model", build_model(spec, params["seed"])),
    ])

    with mlflow.start_run(run_name=spec["name"]) as run:
        mlflow.set_tags({
            "question": spec["question"], "change": spec["change"], "hypothesis": spec["hypothesis"],
            "compare_to": spec.get("compare_to", "-"), "git_commit": git_commit(),
        })
        mlflow.log_params({
            "model": spec["model"], "class_weight": spec.get("class_weight"),
            "dropped_features": ",".join(drop) or "-", "dedup": bool(spec.get("dedup")),
            "n_train": len(X_tr), "n_val": len(X_va), "n_features": X_tr.shape[1],
            "seed": params["seed"], "max_fpr": params["alert"]["max_fpr"],
            "train_hash": int(row_hash(train, feature_columns(train)).sum() % (1 << 62)),
            **{f"hgb_{k}": v for k, v in spec.get("hgb_params", {}).items()},
        })
        mlflow.log_input(mlflow.data.from_pandas(train, source=str(ROOT / "data/processed/train.parquet"),
                                                 name="train", targets=TARGET), context="training")
        mlflow.log_input(mlflow.data.from_pandas(val, source=str(ROOT / "data/processed/val.parquet"),
                                                 name="val", targets=TARGET), context="validation")

        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fit_seconds = time.perf_counter() - t0
        score = pipe.predict_proba(X_va)[:, 1]

        threshold = threshold_for_max_fpr(y_va, score, params["alert"]["max_fpr"])
        at_thr = metrics_at(y_va, score, threshold)
        at_05 = metrics_at(y_va, score, 0.5)
        rank = ranking_metrics(y_va, score)
        mlflow.log_metrics({
            **{f"val_{k}": v for k, v in at_thr.items()},
            **{f"val05_{k}": v for k, v in at_05.items()},
            **{f"val_{k}": v for k, v in rank.items()},
            "threshold": threshold, "fit_seconds": fit_seconds,
        })
        sample = X_tr.head(5)
        mlflow.sklearn.log_model(pipe, name="model", code_paths=[str(ROOT / "src")],
                                 skops_trusted_types=TRUSTED_TYPES,
                                 signature=infer_signature(sample, pipe.predict_proba(sample)[:, 1]),
                                 input_example=sample)

    return {
        "name": spec["name"], "run_id": run.info.run_id, "compare_to": spec.get("compare_to", "-"),
        "threshold": threshold, "fit_seconds": round(fit_seconds, 1), "n_train": len(X_tr),
        **{f"val_{k}": v for k, v in at_thr.items()},
        **{f"val05_{k}": at_05[k] for k in ("recall", "precision", "fpr", "macro_f1")},
        **{f"val_{k}": v for k, v in rank.items()},
    }


def predict_scores(model, df) -> np.ndarray:
    """Probabilidad de ataque; usa solo las columnas con que se entrenó el modelo."""
    return model.predict_proba(df[list(model.feature_names_in_)])[:, 1]
