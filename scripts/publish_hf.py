"""Publica la versión exportada del modelo (models/final) en Hugging Face Hub.

Requiere sesión iniciada: `hf auth login` (token con permiso de escritura).

Uso:
  python scripts/publish_hf.py                         # repo <usuario>/defiende-el-servidor, privado
  python scripts/publish_hf.py --repo usuario/nombre --public

Cargar desde Hugging Face y predecir:
  python -m src.predict --model hf://<usuario>/defiende-el-servidor --input datos.csv --output pred.csv
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import HfApi  # noqa: E402

from src.config import MODEL_DIR, REPORTS, ROOT  # noqa: E402

CARD = """---
license: other
library_name: sklearn
tags: [tabular-classification, intrusion-detection, unsw-nb15, mlflow, sklearn]
datasets: [UNSW-NB15]
metrics: [recall, precision, f1]
---

# Defiende el servidor · detector de ataques de red (UNSW-NB15)

Clasificador binario **normal (0) vs. ataque (1)** sobre conexiones de red de UNSW-NB15.
Proyecto académico de MLOps (Universidad Externado de Colombia).
Código, datos versionados (DVC) y experimentos (MLflow): {github}

## Modelo

| | |
|---|---|
| Algoritmo | `HistGradientBoostingClassifier` (scikit-learn) dentro de un `Pipeline` con el preprocesamiento (log1p + escalado + one-hot) |
| Experimento | `{experiment}` · MLflow run `{run_id}` · registro `{registered_model}` v{version} |
| **Umbral de alerta** | **{threshold:.4f}** — alerta si `predict_proba[:, 1] >= umbral` |
| Criterio | {criterion} |
| Commit | `{git_commit}` |
| Formato | MLflow model (`MLmodel`) serializado con skops |

## Resultados

| Métrica | Validación | Test (evaluado una vez) |
|---|---:|---:|
| Recall (ataques detectados) | {v_recall:.1%} | {t_recall:.1%} |
| Precisión | {v_precision:.1%} | {t_precision:.1%} |
| Tasa de falsos positivos | {v_fpr:.1%} | {t_fpr:.1%} |
| Macro-F1 | {v_f1:.3f} | {t_f1:.3f} |

La tasa de falsas alarmas **sube de 5 % a {t_fpr:.0%} en test** porque el tráfico normal del archivo de
test difiere del de entrenamiento. El umbral debe recalibrarse con tráfico normal de la red donde se use.

## Uso

```python
import sys, mlflow, pandas as pd
from huggingface_hub import snapshot_download

path = snapshot_download("{repo_id}")
sys.path.insert(0, f"{{path}}/code")           # preprocesamiento propio (src/features.py)
model = mlflow.sklearn.load_model(path)
df = pd.read_csv("UNSW_NB15_testing-set.csv", encoding="utf-8-sig")
alerta = model.predict_proba(df[model.feature_names_in_])[:, 1] >= {threshold:.4f}
```

Entrada: las 42 variables de UNSW-NB15 (sin `id`, `label` ni `attack_cat`).

## Limitaciones

- Dataset **sintético de laboratorio (2015)**: no prueba eficacia en una red actual.
- Prevalencia de ataque irreal (55–68 %); en una red real la precisión sería mucho menor.
- Artefactos del laboratorio (TTL de las máquinas atacantes, filas duplicadas).
- Peor detección en **Fuzzers** (79 % detectados en test).

## Datos y atribución

UNSW-NB15 — Moustafa, N. & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion
detection systems*. MilCIS, IEEE. <https://research.unsw.edu.au/projects/unsw-nb15-dataset>.
Uso académico; el modelo hereda las condiciones de uso del dataset.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="usuario/nombre (por defecto <tu usuario>/defiende-el-servidor)")
    ap.add_argument("--public", action="store_true", help="publicar como repo público (por defecto privado)")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()["name"]  # falla con mensaje claro si no hay sesión
    repo_id = args.repo or f"{user}/defiende-el-servidor"
    meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    test = json.loads((REPORTS / "test_metrics.json").read_text(encoding="utf-8"))
    sel = test["selection"]

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "model"
        shutil.copytree(MODEL_DIR, stage)
        shutil.copytree(ROOT / "src", stage / "code" / "src",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (stage / "README.md").write_text(CARD.format(
            github="https://github.com/mjcastano29-tech/defiende-el-servidor", repo_id=repo_id,
            experiment=meta["experiment"], run_id=meta["run_id"], registered_model=meta["registered_model"],
            version=meta["version"], threshold=meta["threshold"], criterion=sel["criterion"],
            git_commit=meta["git_commit"][:7],
            v_recall=sel["val_recall"], v_precision=sel["val_precision"], v_fpr=sel["val_fpr"],
            v_f1=sel["val_macro_f1"], t_recall=test["test"]["recall"], t_precision=test["test"]["precision"],
            t_fpr=test["test"]["fpr"], t_f1=test["test"]["macro_f1"]), encoding="utf-8")

        api.create_repo(repo_id, private=not args.public, exist_ok=True)
        info = api.upload_folder(folder_path=str(stage), repo_id=repo_id,
                                 commit_message=f"{meta['registered_model']} v{meta['version']} "
                                                f"({meta['experiment']}, umbral {meta['threshold']:.4f})")
    print(f"Publicado: https://huggingface.co/{repo_id}  (commit {info.oid[:7]})")


if __name__ == "__main__":
    main()
