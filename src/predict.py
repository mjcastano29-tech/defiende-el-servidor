"""Carga una versión guardada del modelo y genera predicciones.

Ejemplos:
  python -m src.predict --input data/raw/UNSW_NB15_testing-set.csv --output reports/predicciones.csv
  python -m src.predict --model "models:/defiende-el-servidor@champion" --input ... --output ...
  python -m src.predict --model "models:/defiende-el-servidor/1" --input ... --output ...
"""
import argparse
import json

import mlflow
import pandas as pd

from src.config import MODEL_DIR, load_params
from src.data import read_raw
from src.train import predict_scores, setup_mlflow


def load(model_uri: str):
    """Devuelve (modelo, umbral, descripción de la versión)."""
    if model_uri.startswith("models:/"):
        setup_mlflow(load_params())
        name, ref = model_uri.removeprefix("models:/").replace("@", "/").split("/", 1)
        client = mlflow.MlflowClient()
        mv = (client.get_model_version_by_alias(name, ref) if "@" in model_uri
              else client.get_model_version(name, ref))
        return mlflow.sklearn.load_model(model_uri), float(mv.tags["threshold"]), f"{name} v{mv.version}"
    meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    return (mlflow.sklearn.load_model(model_uri), float(meta["threshold"]),
            f"{meta['registered_model']} v{meta['version']} ({model_uri})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(MODEL_DIR), help="ruta local o URI models:/nombre@alias|/versión")
    ap.add_argument("--input", required=True, help="CSV con las columnas de UNSW-NB15")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model, threshold, version = load(args.model)
    df = read_raw(args.input)
    score = predict_scores(model, df)
    out = pd.DataFrame({"prob_ataque": score.round(6), "alerta": (score >= threshold).astype(int)})
    if "id" in df:
        out.insert(0, "id", df["id"])
    out.to_csv(args.output, index=False)
    print(f"Modelo {version} · umbral {threshold:.4f} · {len(out):,} filas · "
          f"{out['alerta'].mean():.1%} alertas → {args.output}")


if __name__ == "__main__":
    main()
