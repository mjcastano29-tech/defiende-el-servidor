"""Preprocesamiento: forma parte del modelo guardado (se aprende solo con train)."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import CATEGORICAL

# MLflow serializa con skops: las funciones propias del modelo deben declararse como confiables
TRUSTED_TYPES = ["src.features.log1p_clip"]


def log1p_clip(X):
    # Variables muy asimétricas (bytes, tasas); no hay negativos, el clip es solo defensa
    return np.log1p(np.clip(X, 0, None))


def build_preprocessor(columns: list[str], min_frequency: float) -> ColumnTransformer:
    cat = [c for c in columns if c in CATEGORICAL]
    num = [c for c in columns if c not in CATEGORICAL]
    numeric = Pipeline([
        ("log1p", FunctionTransformer(log1p_clip, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    # infrequent_if_exist: categorías raras o nunca vistas (p. ej. state=ACC/CLO solo en test)
    # caen en la misma columna "infrecuente" en lugar de romper la predicción
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                sparse_output=False)
    return ColumnTransformer([("num", numeric, num), ("cat", categorical, cat)])
