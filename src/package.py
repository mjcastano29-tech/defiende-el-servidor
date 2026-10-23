"""Empaqueta las versiones v1/v2 a partir de sus runs de MLflow en models/<versión>/.

Cada paquete es lo que se publica en Hugging Face:
  model/          artefacto MLflow copiado tal cual del run (preprocesamiento + estimador, skops)
  config.json     umbral, regla de decisión, variables, métricas de validación y referencias Git/DVC/MLflow
  requirements.txt dependencias fijadas para cargarlo
  README.md       model card

Uso: python -m src.package
"""
import hashlib
import json
import shutil
from pathlib import Path

import yaml
from mlflow import MlflowClient

from src.config import ROOT, load_params
from src.data import CATEGORICAL, FAMILY, ID, TARGET
from src.train import setup_mlflow

MODELS = ROOT / "models"
GITHUB = "https://github.com/mjcastano29-tech/defiende-el-servidor"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(folder: Path) -> str:
    """Huella de una carpeta: sha256 de la lista ordenada (ruta relativa, sha256 del archivo)."""
    lines = [f"{p.relative_to(folder).as_posix()} {sha256(p)}"
             for p in sorted(folder.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def dvc_md5(dvc_file: Path) -> str:
    return yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]


def load_versions() -> dict:
    return yaml.safe_load((ROOT / "configs" / "versions.yaml").read_text(encoding="utf-8"))


def model_artifact_dir(client: MlflowClient, run_id: str) -> Path:
    run = client.get_run(run_id)
    model_id = run.outputs.model_outputs[0].model_id
    uri = client.get_logged_model(model_id).artifact_location
    return Path(uri.removeprefix("file://")), model_id


def build_config(name: str, spec: dict, client: MlflowClient, model_dir: Path, model_id: str) -> dict:
    run = client.get_run(spec["run_id"])
    m, p, t = run.data.metrics, run.data.params, run.data.tags
    data_sha = {f: sha256(ROOT / "data" / "raw" / f)
                for f in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv")}
    stats = json.loads((ROOT / "data" / "processed" / "split_stats.json").read_text())
    sig = json.loads(yaml.safe_load((model_dir / "MLmodel").read_text())["signature"]["inputs"])
    return {
        "version": name,
        "experiment": spec["experiment"],
        "family": spec["family"],
        "summary": " ".join(spec["summary"].split()),
        "change_from_previous": " ".join(spec["change_from_previous"].split()),
        "threshold": m["threshold"],
        "positive_class": "1 = ataque",
        "decision_rule": "y_pred = 1 si score_attack >= threshold, si no 0; score_attack = predict_proba(X)[:, 1]",
        "input_features": [c["name"] for c in sig],
        "categorical_features": [c for c in CATEGORICAL if c in [s["name"] for s in sig]],
        "excluded_columns": [ID, TARGET, FAMILY],
        "model_file": "model/model.skops",
        "model_sha256": sha256(model_dir / "model.skops"),
        "model_artifact_sha256": tree_sha256(model_dir),
        "params": {k: v for k, v in p.items()},
        "validation_metrics": {k.removeprefix("val_"): m[k] for k in
                               ("val_recall", "val_precision", "val_fpr", "val_macro_f1", "val_accuracy",
                                "val_roc_auc", "val_pr_auc", "val_tp", "val_fp", "val_tn", "val_fn")},
        "validation_rows": stats["rows"]["val"],
        "lineage": {
            "git_repository": GITHUB,
            "git_commit_training": t["git_commit"],
            "mlflow_experiment": run.info.experiment_id and client.get_experiment(run.info.experiment_id).name,
            "mlflow_run_id": spec["run_id"],
            "mlflow_model_id": model_id,
            "dvc_data_raw_md5": dvc_md5(ROOT / "data" / "raw.dvc"),
            "dvc_data_processed_md5": dvc_md5(ROOT / "data" / "processed.dvc"),
            "data_sha256": data_sha,
            "split": {"seed": int(p["seed"]), "method": "StratifiedGroupKFold(5) agrupado por filas idénticas, pliegue 0",
                      "rows": stats["rows"]},
        },
    }


CARD = """---
library_name: sklearn
tags: [tabular-classification, intrusion-detection, unsw-nb15, mlflow, scikit-learn]
license: other
---

# Defiende el servidor · {version} · {experiment}

Clasificador binario de conexiones de red **normal (0) / ataque (1)** entrenado con las particiones
publicadas de UNSW-NB15. Proyecto académico de MLOps (Universidad Externado de Colombia, 2026-II).

## Propósito

Priorizar qué conexiones merecen una alerta para un equipo de seguridad: maximizar los ataques
detectados sin alarmar más del 5 % del tráfico normal **en validación**. Es un ejercicio de versionado
y trazabilidad; no está validado para una red real.

## Versión

{summary}

**Cambio respecto a la versión anterior:** {change}

| | |
|---|---|
| Familia | {family} |
| Umbral | **{threshold:.6f}** — `y_pred = 1` si `predict_proba[:, 1] >= umbral` |
| Clase positiva | 1 = ataque |
| SHA-256 de `model/model.skops` | `{model_sha}` |
| Huella de la carpeta `model/` | `{tree_sha}` |

## Datos

- UNSW-NB15, `UNSW_NB15_training-set.csv` (175.341 filas, SHA-256 `{sha_train}`) y
  `UNSW_NB15_testing-set.csv` (82.332 filas, SHA-256 `{sha_test}`).
- Entrenamiento ({n_train} filas) y validación ({n_val} filas) salen solo del archivo de entrenamiento,
  con `StratifiedGroupKFold` agrupado por filas de variables idénticas (ningún grupo repetido queda a
  ambos lados), semilla {seed}.
- Entradas: 42 variables; se excluyen `id`, `label` y `attack_cat`.
- Preprocesamiento ajustado solo con entrenamiento: `log1p` + `StandardScaler` en numéricas, `OneHotEncoder`
  (categorías raras < 0,1 % e inéditas → "infrecuente") en `proto`, `service`, `state`.

## Métricas de validación

Con el umbral de la versión ({n_val} filas; clase positiva = ataque):

| Recall ataque | Precisión ataque | Tasa de falsos positivos | Macro-F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| {recall:.4f} | {precision:.4f} | {fpr:.4f} | {f1:.4f} | {auc:.4f} |

Matriz de confusión (filas = real, columnas = predicho): normal → [{tn} | {fp}], ataque → [{fn} | {tp}].
Denominadores cero: la métrica se reporta como 0.
{test_section}
## Uso

```bash
pip install -r requirements.txt huggingface_hub
```

```python
import json, mlflow, pandas as pd
from huggingface_hub import snapshot_download

path = snapshot_download("{hf_repo}", revision="<SHA completo de la revisión>")
cfg = json.load(open(f"{{path}}/config.json"))
model = mlflow.sklearn.load_model(f"{{path}}/model")          # preprocesamiento + estimador
X = pd.read_csv("UNSW_NB15_testing-set.csv", encoding="utf-8-sig")
score = model.predict_proba(X[cfg["input_features"]])[:, 1]
y_pred = (score >= cfg["threshold"]).astype(int)
```

O desde el repositorio de código: `python -m src.predict --model hf://{hf_repo}@<SHA> --input ... --output ...`

## Trazabilidad

| Pieza | Referencia |
|---|---|
| Código | {github} · tag `{version}` · commit de entrenamiento `{commit}` |
| Datos (DVC) | `data/raw` md5 `{dvc_raw}` · `data/processed` md5 `{dvc_proc}` · remoto `https://huggingface.co/datasets/Marimar2006/defiende-el-servidor-dvc` |
| Experimento (MLflow) | run `{run_id}` · modelo `{model_id}` |

## Limitaciones

- Dataset **sintético de laboratorio (2015)**; no demuestra eficacia sobre una red actual.
- Prevalencia de ataque irreal (68 % en entrenamiento, 55 % en test): la precisión sería mucho menor en una red
  donde el ataque es raro.
- Artefactos del laboratorio: el TTL identifica las máquinas atacantes; 42 % de filas repetidas en entrenamiento y
  229 grupos de entradas idénticas con etiqueta contradictoria.
- El umbral se calibró con el tráfico normal de validación; si el tráfico normal cambia, la tasa de falsas alarmas
  cambia (ver resultado de test en v2).

## Referencias

Moustafa, N. & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems*.
MilCIS, IEEE. <https://research.unsw.edu.au/projects/unsw-nb15-dataset>. El modelo hereda las condiciones de uso
académico del dataset.
"""


def render_card(cfg: dict, hf_repo: str, test_section: str = "") -> str:
    v, lin = cfg["validation_metrics"], cfg["lineage"]
    return CARD.format(
        version=cfg["version"], experiment=cfg["experiment"], summary=cfg["summary"],
        change=cfg["change_from_previous"], family=cfg["family"], threshold=cfg["threshold"],
        model_sha=cfg["model_sha256"], tree_sha=cfg["model_artifact_sha256"],
        sha_train=lin["data_sha256"]["UNSW_NB15_training-set.csv"],
        sha_test=lin["data_sha256"]["UNSW_NB15_testing-set.csv"],
        n_train=lin["split"]["rows"]["train"], n_val=lin["split"]["rows"]["val"], seed=lin["split"]["seed"],
        recall=v["recall"], precision=v["precision"], fpr=v["fpr"], f1=v["macro_f1"], auc=v["roc_auc"],
        tn=int(v["tn"]), fp=int(v["fp"]), fn=int(v["fn"]), tp=int(v["tp"]), test_section=test_section,
        hf_repo=hf_repo, github=GITHUB, commit=lin["git_commit_training"],
        dvc_raw=lin["dvc_data_raw_md5"], dvc_proc=lin["dvc_data_processed_md5"],
        run_id=lin["mlflow_run_id"], model_id=lin["mlflow_model_id"])


def package_all() -> dict:
    setup_mlflow(load_params())
    client = MlflowClient()
    versions = load_versions()
    out = {}
    for name, spec in versions["versions"].items():
        src_dir, model_id = model_artifact_dir(client, spec["run_id"])
        dest = MODELS / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src_dir, dest / "model",
                        ignore=shutil.ignore_patterns("registered_model_meta", "__pycache__"))
        cfg = build_config(name, spec, client, dest / "model", model_id)
        (dest / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.copy(dest / "model" / "requirements.txt", dest / "requirements.txt")
        (dest / "README.md").write_text(render_card(cfg, versions["hf_repo"]), encoding="utf-8")
        out[name] = cfg
        print(f"{name}: {spec['experiment']} · umbral {cfg['threshold']:.4f} · model.skops {cfg['model_sha256'][:12]}…")
    return out


if __name__ == "__main__":
    package_all()
