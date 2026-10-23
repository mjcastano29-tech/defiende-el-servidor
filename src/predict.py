"""Carga una versión empaquetada del modelo y genera predicciones.

Ejemplos:
  python -m src.predict --model models/v2 --input data/raw/UNSW_NB15_testing-set.csv --output pred.csv
  python -m src.predict --model hf://<usuario>/defiende-el-servidor@<SHA> --input entrada.csv --output pred.csv
  python -m src.predict --model "models:/defiende-el-servidor@champion" --input ... --output ...

Salida: row_id, [y_true], score_attack, y_pred  (y_pred = 1 si score_attack >= umbral de la versión).
"""
import argparse
import json
import os
from pathlib import Path

import mlflow
import pandas as pd

from src.config import ROOT, load_params
from src.data import ID, TARGET, read_raw
from src.train import predict_scores, setup_mlflow


def download_hf(model_ref: str) -> Path:
    """Descarga hf://usuario/repo@SHA a una carpeta con archivos reales (no enlaces simbólicos a la caché:
    MLflow rechaza cargar modelos cuyos archivos apuntan fuera de su carpeta)."""
    from huggingface_hub import snapshot_download
    repo, _, revision = model_ref.removeprefix("hf://").partition("@")
    base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "defiende-local"
    target = base / repo.replace("/", "--") / (revision or "main")
    return Path(snapshot_download(repo, revision=revision or None, local_dir=target))


def load(model_ref: str):
    """Devuelve (modelo, umbral, descripción). Acepta carpeta de versión, hf://repo@SHA o models:/."""
    if model_ref.startswith("models:/"):
        setup_mlflow(load_params())
        name, ref = model_ref.removeprefix("models:/").replace("@", "/").split("/", 1)
        client = mlflow.MlflowClient()
        mv = (client.get_model_version_by_alias(name, ref) if "@" in model_ref
              else client.get_model_version(name, ref))
        return mlflow.sklearn.load_model(model_ref), float(mv.tags["threshold"]), f"{name} v{mv.version}"
    if model_ref.startswith("hf://"):
        folder = download_hf(model_ref)
        label = model_ref
    else:
        folder = Path(model_ref)
        label = str(folder)
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model = mlflow.sklearn.load_model(str(folder / "model"))
    return model, float(cfg["threshold"]), f"{cfg['version']} ({cfg['experiment']}) · {label}"


def predict_frame(model, threshold: float, df: pd.DataFrame, source: str) -> pd.DataFrame:
    score = predict_scores(model, df)
    ids = df[ID] if ID in df else pd.Series(range(1, len(df) + 1))
    out = pd.DataFrame({"row_id": [f"{source}#id={i}" for i in ids]})
    if TARGET in df:
        out["y_true"] = df[TARGET].to_numpy()
    out["score_attack"] = score
    out["y_pred"] = (score >= threshold).astype(int)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "v2"),
                    help="carpeta de versión (models/v1, models/v2), hf://usuario/repo@SHA o models:/nombre@alias")
    ap.add_argument("--input", required=True, help="CSV con las columnas de UNSW-NB15")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model, threshold, version = load(args.model)
    out = predict_frame(model, threshold, read_raw(args.input), Path(args.input).name)
    out.to_csv(args.output, index=False, float_format="%.10f")
    print(f"Modelo {version} · umbral {threshold:.6f} · {len(out):,} filas · "
          f"{out['y_pred'].mean():.1%} alertas → {args.output}")


if __name__ == "__main__":
    main()
