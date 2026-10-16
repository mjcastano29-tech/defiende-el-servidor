"""Lectura, validación y partición de UNSW-NB15."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import ROOT

ID, TARGET, FAMILY = "id", "label", "attack_cat"
CATEGORICAL = ["proto", "service", "state"]


class DataValidationError(ValueError):
    """Los datos no cumplen lo que el resto del pipeline asume."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw(path: Path) -> pd.DataFrame:
    # utf-8-sig: el CSV trae BOM; sin esto la primera columna se llama '﻿id'
    return pd.read_csv(path, encoding="utf-8-sig")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Variables disponibles al predecir: todo menos el identificador y las respuestas."""
    return [c for c in df.columns if c not in (ID, TARGET, FAMILY)]


def row_hash(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Hash por fila de las variables de entrada: filas idénticas -> mismo hash."""
    return pd.util.hash_pandas_object(df[cols], index=False)


def validate(df: pd.DataFrame, name: str, expected_rows: int) -> None:
    problems = []
    if len(df) != expected_rows:
        problems.append(f"{len(df)} filas, se esperaban {expected_rows} (¿train/test intercambiados?)")
    missing = {ID, TARGET, FAMILY, *CATEGORICAL} - set(df.columns)
    if missing:
        problems.append(f"faltan columnas {sorted(missing)}")
    else:
        if df[TARGET].isna().any() or not set(df[TARGET].unique()) <= {0, 1}:
            problems.append("label tiene valores distintos de 0/1")
        if not (df[FAMILY].eq("Normal") == df[TARGET].eq(0)).all():
            problems.append("attack_cat y label no son consistentes")
    if problems:
        raise DataValidationError(f"{name}: " + "; ".join(problems))


def split_indices(train: pd.DataFrame, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Posiciones (0-based) de las filas del train oficial que van a entrenamiento y a validación.

    Agrupa por filas con variables predictoras idénticas: ningún grupo queda repartido."""
    groups = row_hash(train, feature_columns(train))
    splitter = StratifiedGroupKFold(n_splits=params["split"]["val_folds"], shuffle=True,
                                    random_state=params["seed"])
    tr_idx, va_idx = next(splitter.split(train, train[TARGET], groups))
    shared = set(groups.iloc[tr_idx]) & set(groups.iloc[va_idx])
    if shared:
        raise DataValidationError(f"{len(shared)} grupos de filas idénticas cruzan train/validación")
    return tr_idx, va_idx


def prepare(params: dict) -> dict:
    """Valida los CSV, separa train/validación agrupando copias y guarda parquet."""
    cfg = params["data"]
    out_dir = ROOT / cfg["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    train = read_raw(ROOT / cfg["train"])
    test = read_raw(ROOT / cfg["test"])
    validate(train, "train", cfg["expected_rows"]["train"])
    validate(test, "test", cfg["expected_rows"]["test"])

    feats = feature_columns(train)
    groups = row_hash(train, feats)
    tr_idx, va_idx = split_indices(train, params)

    keep = feats + [TARGET, FAMILY]  # se descarta id: es el número de fila de un archivo ordenado
    tr, va, te = train.iloc[tr_idx][keep], train.iloc[va_idx][keep], test[keep].copy()
    # Filas del test idénticas a alguna de train: se reportan aparte para medir cuánto inflan
    te["seen_in_train"] = row_hash(te, feats).isin(set(groups)).to_numpy()

    for name, df in [("train", tr), ("val", va), ("test", te)]:
        df.reset_index(drop=True).to_parquet(out_dir / f"{name}.parquet", index=False)

    stats = {
        "source_sha256": {k: sha256(ROOT / cfg[k]) for k in ("train", "test")},
        "rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "attack_rate": {k: round(float(d[TARGET].mean()), 4) for k, d in [("train", tr), ("val", va), ("test", te)]},
        "duplicate_rows_train_split": int(row_hash(tr, feats).duplicated().sum()),
        "test_seen_in_train": int(te["seen_in_train"].sum()),
        "n_features": len(feats),
    }
    (out_dir / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_split(params: dict, name: str) -> pd.DataFrame:
    path = ROOT / params["data"]["processed_dir"] / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} no existe: ejecuta primero la etapa de preparación")
    return pd.read_parquet(path)


def xy(df: pd.DataFrame, drop: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    cols = [c for c in feature_columns(df) if c not in (drop or []) and c != "seen_in_train"]
    return df[cols], df[TARGET].to_numpy()
