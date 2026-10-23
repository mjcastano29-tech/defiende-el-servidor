"""Publica las versiones del modelo en Hugging Face Hub y registra sus revisiones (SHA completo).

Requiere sesión: `hf auth login` (token de escritura; nunca se guarda en el repo).

  python scripts/publish_hf.py versions     # sube v1 y luego v2 (cada una es un commit + tag)
  python scripts/publish_hf.py test-card    # nueva revisión: solo la model card de v2 con el test

Escribe configs/hf_revisions.json con repo, SHA, tag y huellas de cada revisión.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import HfApi  # noqa: E402

from src.config import ROOT  # noqa: E402
from src.package import load_versions, render_card  # noqa: E402

REVISIONS = ROOT / "configs" / "hf_revisions.json"


def repo_id(api: HfApi) -> str:
    return f"{api.whoami()['name']}/{load_versions()['hf_repo']}"


def read_revisions() -> dict:
    return json.loads(REVISIONS.read_text(encoding="utf-8")) if REVISIONS.exists() else {}


def write_revisions(data: dict) -> None:
    REVISIONS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def publish_versions(api: HfApi) -> None:
    repo = repo_id(api)
    api.create_repo(repo, private=False, exist_ok=True)
    revisions = read_revisions()
    for name in load_versions()["versions"]:
        folder = ROOT / "models" / name
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        card = render_card(cfg, repo)
        # Cada versión reemplaza el contenido completo del repo: la revisión contiene solo esa versión
        api.upload_folder(folder_path=str(folder), repo_id=repo, delete_patterns=["*"],
                          ignore_patterns=["README.md", "**/__pycache__/**"],
                          commit_message=f"{name}: {cfg['experiment']} (run {cfg['lineage']['mlflow_run_id']}, "
                                         f"umbral {cfg['threshold']:.6f})")
        info = api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=repo,
                               commit_message=f"{name}: model card")
        sha = info.oid
        api.create_tag(repo, tag=name, revision=sha, tag_message=f"{name}: {cfg['experiment']}", exist_ok=False)
        revisions[name] = {"repo": repo, "url": f"https://huggingface.co/{repo}/tree/{sha}", "sha": sha, "tag": name,
                           "git_tag": name, "experiment": cfg["experiment"],
                           "mlflow_run_id": cfg["lineage"]["mlflow_run_id"], "threshold": cfg["threshold"],
                           "model_sha256": cfg["model_sha256"], "model_artifact_sha256": cfg["model_artifact_sha256"]}
        print(f"{name}: https://huggingface.co/{repo}/tree/{sha}")
    write_revisions(revisions)


def publish_test_card(api: HfApi) -> None:
    repo = repo_id(api)
    revisions = read_revisions()
    evaluated = revisions["v2"]
    test = json.loads((ROOT / "evidencias" / "metricas-test.json").read_text(encoding="utf-8"))
    assert test["model_sha256"] == evaluated["model_sha256"], "el modelo evaluado no es el publicado como v2"
    m, cm = test["metricas"], test["matriz_confusion"]["valores"]
    sub = test["subconjunto_sin_entradas_vistas_en_train"]
    section = f"""
## Resultado final en test (evaluado una sola vez)

Revisión evaluada: `{evaluated['sha']}` (tag `v2`) · SHA-256 de `model/model.skops`: `{test['model_sha256']}`.
Esta revisión solo actualiza la documentación; los archivos del modelo son idénticos a los de la revisión evaluada.

`UNSW_NB15_testing-set.csv` completo ({test['filas_evaluadas']:,} filas, SHA-256 `{test['dataset_sha256']}`), umbral {test['umbral']:.6f}:

| Recall ataque | Precisión ataque | Tasa de falsos positivos | Macro-F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| {m['recall']:.4f} | {m['precision']:.4f} | {m['fpr']:.4f} | {m['macro_f1']:.4f} | {m['roc_auc']:.4f} |

Matriz de confusión: normal → [{cm[0][0]} | {cm[0][1]}], ataque → [{cm[1][0]} | {cm[1][1]}].
Sin las {test['filas_evaluadas'] - sub['filas']:,} filas cuyas entradas también están en entrenamiento:
recall {sub['recall']:.4f}, FPR {sub['fpr']:.4f}.

**La tasa de falsas alarmas sube de 5 % (validación) a {m['fpr']:.1%} (test)** porque el tráfico normal del archivo
de test difiere del de entrenamiento (41 % con `sttl=254` frente a 20 %). La capacidad de ordenar se mantiene
(ROC-AUC {m['roc_auc']:.3f}); el umbral debe recalibrarse con tráfico normal de la red donde se use.
"""
    cfg = json.loads((ROOT / "models" / "v2" / "config.json").read_text(encoding="utf-8"))
    card = render_card(cfg, repo, test_section=section)
    info = api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=repo,
                           revision="main", commit_message="v2: añadir resultado final de test a la model card")
    api.create_tag(repo, tag="v2-test-card", revision=info.oid, exist_ok=False)
    revisions["v2_model_card_test"] = {"repo": repo, "sha": info.oid, "tag": "v2-test-card",
                                       "url": f"https://huggingface.co/{repo}/tree/{info.oid}",
                                       "evaluated_revision": evaluated["sha"],
                                       "model_sha256": test["model_sha256"]}
    write_revisions(revisions)
    print(f"model card con test: https://huggingface.co/{repo}/tree/{info.oid}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["versions", "test-card"])
    step = ap.parse_args().step
    api = HfApi()
    publish_versions(api) if step == "versions" else publish_test_card(api)
