"""Genera la carpeta evidencias/ que pide el taller a partir de datos, MLflow y los paquetes v1/v2.

  auditoria.json          fuentes, huellas, esquema, clases, faltantes, duplicados, splits y cruces
  asignacion_splits.csv   row_id -> train/val/test (asignación reproducible)
  experimentos.csv        una fila por configuración (runs canónicos)
  seleccion.json          versión elegida, referencias y evidencia de congelación previa al test
  metricas-test.json      métricas recalculadas desde predicciones-test.csv
  predicciones-test.csv   row_id, y_true, score_attack, y_pred para las 82.332 filas del test
  mlflow/                 exportación de los runs (JSON + CSV + MLmodel)
  recuperacion/           20 filas fijas de validación y salidas originales de v1/v2

Uso: python scripts/build_evidence.py
"""
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from src.config import ROOT, load_params  # noqa: E402
from src.data import FAMILY, ID, TARGET, feature_columns, read_raw, row_hash, split_indices  # noqa: E402
from src.metrics import metrics_at, ranking_metrics  # noqa: E402
from src.package import dvc_md5, load_versions, sha256  # noqa: E402
from src.predict import load, predict_frame  # noqa: E402
from src.train import setup_mlflow  # noqa: E402

EV = ROOT / "evidencias"
RAW = ROOT / "data" / "raw"
TRAIN_CSV, TEST_CSV = "UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"
CANONICAL_COMMIT = "80cdc99"     # commit cuyo pipeline produjo los runs entregados (2026-10-14)
REPRO_COMMIT = "450942b"         # re-ejecución completa (2026-10-16): mismas métricas
ANNEX_SHA256 = {TRAIN_CSV: "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
                TEST_CSV: "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559"}
FAMILIES = {"dummy": "referencia (DummyClassifier)", "logreg": "lineal (regresión logística)",
            "hgb": "árboles (HistGradientBoosting)"}
CONCLUSIONS = {
    "E0_referencia": "Confirmada: piso sin capacidad; con FPR<=5 % no detecta ataques y con 0,5 lo marca todo.",
    "E1_logreg": "Confirmada: ROC-AUC 0,986; recall 90,5 % con FPR 5 %. Base fuerte y barata → v1.",
    "E2_logreg_balanceada": "Confirmada: balancear solo desplaza el umbral; con el umbral ajustado no aporta (90,4 %).",
    "E3_hgb": "Confirmada: +4,2 puntos de recall frente a E1 con la misma FPR; es el cambio que más aporta.",
    "E4_hgb_sin_ttl": "Confirmada en validación: sin TTL pierde solo 0,4 puntos; no depende del atajo del laboratorio.",
    "E5_hgb_sin_duplicados": "Confirmada: con 42 % menos filas pierde 0,4 puntos; las copias aportan poco.",
    "E6_hgb_complejo": "Confirmada: +0,5 puntos frente a E3 a 3,8x el tiempo (segundos). Elegida por el criterio → v2.",
}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def jdump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def file_profile(name: str) -> dict:
    df = read_raw(RAW / name)
    return {
        "archivo": name, "sha256": sha256(RAW / name), "sha256_anexo_taller": ANNEX_SHA256[name],
        "coincide_con_anexo": sha256(RAW / name) == ANNEX_SHA256[name],
        "bytes": (RAW / name).stat().st_size, "filas": len(df), "columnas": df.shape[1],
        "encoding": "utf-8 con BOM (leer con utf-8-sig)",
        "tipos": {c: str(t) for c, t in df.dtypes.items()},
        "clases_label": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "tasa_ataque": round(float(df[TARGET].mean()), 6),
        "attack_cat": {k: int(v) for k, v in df[FAMILY].value_counts().items()},
        "faltantes_nan_total": int(df.isna().sum().sum()),
        "faltantes_codificados": {"service == '-'": int((df["service"] == "-").sum())},
        "id_unico_y_consecutivo": bool((df[ID] == np.arange(1, len(df) + 1)).all()),
    }


def duplicates(df: pd.DataFrame) -> dict:
    feats = feature_columns(df)
    h = row_hash(df, feats)
    labels = df.assign(_h=h).groupby("_h")[TARGET].nunique()
    conflict = labels[labels > 1].index
    return {
        "criterio": "filas con las 42 variables predictoras idénticas (sin id, label ni attack_cat)",
        "filas_duplicadas": int(h.duplicated().sum()),
        "pct_duplicadas": round(float(h.duplicated().mean() * 100), 2),
        "entradas_unicas": int(h.nunique()),
        "grupos_con_etiqueta_contradictoria": int(len(conflict)),
        "filas_en_grupos_contradictorios": int(h.isin(conflict).sum()),
    }


def build_audit(params: dict) -> pd.DataFrame:
    train, test = read_raw(RAW / TRAIN_CSV), read_raw(RAW / TEST_CSV)
    feats = feature_columns(train)
    tr_idx, va_idx = split_indices(train, params)
    split = np.array(["train"] * len(train), dtype=object)
    split[va_idx] = "val"

    assign = pd.concat([
        pd.DataFrame({"row_id": [f"{TRAIN_CSV}#id={i}" for i in train[ID]], "split": split}),
        pd.DataFrame({"row_id": [f"{TEST_CSV}#id={i}" for i in test[ID]], "split": "test"}),
    ])
    assign.to_csv(EV / "asignacion_splits.csv", index=False)

    # Comprobación: la asignación recalculada reproduce los parquet versionados con DVC
    proc = ROOT / params["data"]["processed_dir"]
    for name, idx in [("train", tr_idx), ("val", va_idx)]:
        stored = pd.read_parquet(proc / f"{name}.parquet")
        recalc = train.iloc[idx][feats + [TARGET, FAMILY]].reset_index(drop=True)
        assert stored.drop(columns=[], errors="ignore").equals(recalc), f"{name} no coincide con el parquet"

    h_tr, h_va, h_te = (row_hash(train.iloc[tr_idx], feats), row_hash(train.iloc[va_idx], feats),
                        row_hash(test, feats))
    split_hash = lambda ids: sha256_str("\n".join(map(str, sorted(ids))))  # noqa: E731
    audit = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fuentes": {
            "pagina_oficial": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
            "descarga_oficial": "SharePoint UNSW, carpeta 'UNSW-NB15 dataset/CSV Files/Training and Testing Sets' "
                                "(última modificación 2024-02-09), API REST anónima",
            "mirror_docente": "https://raw.githubusercontent.com/Nir-J/ML-Projects/"
                              "44bbd02b9e70f74dd915218d5eb702f125341ffc/UNSW-Network_Packet_Classification/",
            "fecha_descarga": "2026-10-02",
            "verificacion": "Mirror y descarga oficial idénticos byte a byte (mismo SHA-256); coinciden con el anexo.",
            "atribucion": "Moustafa, N. & Slay, J. (2015). UNSW-NB15. MilCIS, IEEE. Uso académico.",
        },
        "archivos": [file_profile(TRAIN_CSV), file_profile(TEST_CSV)],
        "duplicados": {"train": duplicates(train), "test": duplicates(test)},
        "politica_limpieza": [
            "No se elimina ninguna fila ni del entrenamiento ni del test (población completa).",
            "Duplicados y grupos contradictorios se conservan; se impide su fuga agrupando el split por entradas idénticas.",
            "El experimento E5 mide el efecto de deduplicar el split de entrenamiento (sin cambiar validación).",
            "service='-' (servicio no identificado) se trata como categoría propia, no como faltante a imputar.",
            "Categorías raras (<0,1 %) o no vistas (p. ej. state ACC/CLO solo en test) → columna 'infrecuente'.",
        ],
        "variables": {
            "usadas": feats, "n_usadas": len(feats),
            "categoricas": ["proto", "service", "state"],
            "excluidas": {"label": "respuesta", "attack_cat": "revela la respuesta (Normal <=> label 0)",
                          "id": "número de fila de un archivo ordenado por clase; solo para auditoría y row_id"},
        },
        "particiones": {
            "semilla": params["seed"],
            "metodo": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) sobre el archivo de "
                      "entrenamiento; grupos = hash de las 42 variables; se usa el primer pliegue como validación",
            "tamanos": {"train": len(tr_idx), "val": len(va_idx), "test": len(test)},
            "tasa_ataque": {"train": round(float(train.iloc[tr_idx][TARGET].mean()), 4),
                            "val": round(float(train.iloc[va_idx][TARGET].mean()), 4),
                            "test": round(float(test[TARGET].mean()), 4)},
            "huellas_asignacion_sha256": {"train": split_hash(train.iloc[tr_idx][ID]),
                                          "val": split_hash(train.iloc[va_idx][ID])},
            "huellas_parquet_sha256": {n: sha256(proc / f"{n}.parquet") for n in ("train", "val", "test")},
            "dvc": {"data/raw.dvc": dvc_md5(ROOT / "data/raw.dvc"),
                    "data/processed.dvc": dvc_md5(ROOT / "data/processed.dvc")},
            "asignacion_por_fila": "evidencias/asignacion_splits.csv",
            "transformacion": "src/data.py:prepare — valida filas/columnas, descarta id, separa train/val, "
                              "marca filas de test vistas en train y escribe parquet",
        },
        "cruces_entre_particiones": {
            "grupos_compartidos_train_val": len(set(h_tr) & set(h_va)),
            "filas_test_con_entrada_en_train": int(h_te.isin(set(h_tr)).sum()),
            "filas_test_con_entrada_en_val": int(h_te.isin(set(h_va)).sum()),
            "filas_test_con_entrada_en_train_o_val": int(h_te.isin(set(h_tr) | set(h_va)).sum()),
            "pct_test_solapado": round(float(h_te.isin(set(h_tr) | set(h_va)).mean() * 100), 2),
            "categorias_test_no_vistas_en_train": {c: sorted(set(test[c]) - set(train[c]))
                                                   for c in ("proto", "service", "state")},
            "nota": "El test no se modifica; el solapamiento se reporta y se mide su efecto en metricas-test.json.",
        },
    }
    jdump(audit, EV / "auditoria.json")
    return test


def sha256_str(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()


def runs_by_commit(client: MlflowClient, params: dict) -> pd.DataFrame:
    exp = client.get_experiment_by_name(params["mlflow"]["experiment"])
    rows = []
    for r in client.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"]):
        rows.append({"run": r, "name": r.data.tags["mlflow.runName"], "commit": r.data.tags.get("git_commit", "")})
    return pd.DataFrame(rows)


def export_mlflow(client: MlflowClient, runs: pd.DataFrame):
    out = EV / "mlflow"
    if out.exists():
        shutil.rmtree(out)
    (out / "runs").mkdir(parents=True)
    flat = []
    for _, row in runs.iterrows():
        r = row["run"]
        rid = r.info.run_id
        model_id = r.outputs.model_outputs[0].model_id if r.outputs and r.outputs.model_outputs else None
        record = {
            "run_id": rid, "run_name": row["name"], "experiment_id": r.info.experiment_id,
            "status": r.info.status,
            "start_time_utc": datetime.fromtimestamp(r.info.start_time / 1000, timezone.utc).isoformat(),
            "end_time_utc": datetime.fromtimestamp(r.info.end_time / 1000, timezone.utc).isoformat(),
            "conjunto": "canonico" if row["commit"].startswith(CANONICAL_COMMIT) else "reproduccion",
            "params": dict(r.data.params), "metrics": dict(r.data.metrics),
            "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.log-model")},
            "inputs": [{"name": d.dataset.name, "digest": d.dataset.digest, "source": d.dataset.source,
                        "context": [t.value for t in d.tags]} for d in r.inputs.dataset_inputs],
            "logged_model_id": model_id,
        }
        jdump(record, out / "runs" / f"{rid}.json")
        if model_id:
            art = Path(client.get_logged_model(model_id).artifact_location.removeprefix("file://"))
            dest = out / "models" / f"{row['name']}__{rid}"
            dest.mkdir(parents=True)
            for f in ("MLmodel", "requirements.txt", "python_env.yaml", "input_example.json"):
                shutil.copy(art / f, dest / f)
            (dest / "model.skops.sha256").write_text(sha256(art / "model.skops") + "  model.skops\n")
        flat.append({"run_id": rid, "run_name": row["name"], "conjunto": record["conjunto"],
                     "git_commit": row["commit"], "status": r.info.status, "start_time_utc": record["start_time_utc"],
                     **{f"param.{k}": v for k, v in r.data.params.items()},
                     **{f"metric.{k}": v for k, v in r.data.metrics.items()},
                     **{f"tag.{k}": r.data.tags.get(k) for k in ("question", "hypothesis", "change", "compare_to",
                                                                 "selected")}})
    pd.DataFrame(flat).to_csv(out / "runs.csv", index=False)
    (out / "LEEME.md").write_text(
        "# Exportación de MLflow\n\n"
        "- `runs.csv`: una fila por run (parámetros, métricas y tags). `conjunto=canonico` son los runs del commit "
        f"`{CANONICAL_COMMIT}` usados en la entrega; `reproduccion` es la re-ejecución del commit `{REPRO_COMMIT}` "
        "con métricas idénticas.\n"
        "- `runs/<run_id>.json`: registro completo de cada run (incluye datasets de entrada con su digest).\n"
        "- `models/<experimento>__<run_id>/`: metadatos del modelo logueado y SHA-256 de `model.skops`.\n\n"
        "Consulta sin servidor:\n\n```python\nimport pandas as pd\n"
        "runs = pd.read_csv('evidencias/mlflow/runs.csv')\n"
        "runs[runs.conjunto == 'canonico'][['run_name', 'run_id', 'metric.val_recall', 'metric.val_fpr']]\n```\n\n"
        "La base completa (`mlflow/mlflow.db` + artefactos) se recupera con `dvc pull` y se abre con\n"
        "`mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`.\n", encoding="utf-8")


def build_experiments(runs: pd.DataFrame, params: dict, versions: dict) -> pd.DataFrame:
    specs = {s["name"]: s for s in params["experiments"]}
    version_of = {v["run_id"]: k for k, v in versions["versions"].items()}
    raw_ref = yaml.safe_load(git("show", f"{CANONICAL_COMMIT}:data/raw.dvc"))["outs"][0]["md5"]
    rows = []
    for _, row in runs[runs["commit"].str.startswith(CANONICAL_COMMIT)].iterrows():
        r, s = row["run"], specs[row["name"]]
        m, p = r.data.metrics, r.data.params
        rows.append({
            "id": row["name"].split("_")[0], "experimento": row["name"], "familia": FAMILIES[s["model"]],
            "compara_con": s.get("compare_to", "-"), "pregunta": s["question"],
            "hipotesis": " ".join(s["hypothesis"].split()), "cambio": s["change"],
            "parametros": json.dumps({k: v for k, v in p.items() if k not in ("seed", "max_fpr")}, ensure_ascii=False),
            "semilla": p["seed"], "commit_git": row["commit"],
            "dvc_raw_md5": raw_ref, "dvc_processed_md5": dvc_md5(ROOT / "data/processed.dvc"),
            "run_id": r.info.run_id, "version_entregada": version_of.get(r.info.run_id, ""),
            "umbral": m["threshold"], "recall": m["val_recall"], "precision": m["val_precision"],
            "fpr": m["val_fpr"], "macro_f1": m["val_macro_f1"], "accuracy": m["val_accuracy"],
            "roc_auc": m["val_roc_auc"], "tp": int(m["val_tp"]), "fp": int(m["val_fp"]),
            "tn": int(m["val_tn"]), "fn": int(m["val_fn"]),
            "recall_umbral_0.5": m["val05_recall"], "fpr_umbral_0.5": m["val05_fpr"],
            "filas_entrenamiento": int(p["n_train"]), "filas_validacion": int(p["n_val"]),
            "segundos_entrenamiento": round(m["fit_seconds"], 1),
            "conclusion_validacion": CONCLUSIONS[row["name"]],
        })
    df = pd.DataFrame(rows)
    df.to_csv(EV / "experimentos.csv", index=False)
    return df


def prefect_freeze_evidence() -> dict:
    db = Path.home() / ".prefect" / "prefect.db"
    if not db.exists():
        return {"nota": "base local de Prefect no disponible"}
    con = sqlite3.connect(db)
    fr = con.execute("SELECT id FROM flow_run WHERE name='tangible-tody'").fetchone()
    tasks = {n.rsplit("-", 1)[0]: {"inicio_utc": s, "fin_utc": e, "estado": st} for n, st, s, e in con.execute(
        "SELECT name, state_name, start_time, end_time FROM task_run WHERE flow_run_id=? AND "
        "(name LIKE 'comparar%' OR name LIKE 'evaluar%')", (fr[0],))}
    return {"flow_run_prefect": "tangible-tody", "tareas": tasks}


def build_test(test: pd.DataFrame, versions: dict, exps: pd.DataFrame):
    v2 = json.loads((ROOT / "models/v2/config.json").read_text(encoding="utf-8"))
    model, thr, desc = load(str(ROOT / "models/v2"))
    pred = predict_frame(model, thr, test, TEST_CSV)
    pred.to_csv(EV / "predicciones-test.csv", index=False, float_format="%.10f")

    # Recalcular desde el CSV guardado (lo que hará el docente)
    back = pd.read_csv(EV / "predicciones-test.csv")
    y, s = back["y_true"].to_numpy(), back["score_attack"].to_numpy()
    assert (back["y_pred"].to_numpy() == (s >= thr).astype(int)).all()
    seen = (row_hash(test, feature_columns(test)).isin(
        set(row_hash(read_raw(RAW / TRAIN_CSV), feature_columns(test))))).to_numpy()
    m = metrics_at(y, s, thr)
    metrics = {
        "version": "v2", "experimento": v2["experiment"], "run_id": v2["lineage"]["mlflow_run_id"],
        "model_sha256": v2["model_sha256"], "model_artifact_sha256": v2["model_artifact_sha256"],
        "dataset": TEST_CSV, "dataset_sha256": sha256(RAW / TEST_CSV),
        "predicciones_sha256": sha256(EV / "predicciones-test.csv"),
        "filas_evaluadas": len(back), "umbral": thr, "clase_positiva": "1 = ataque",
        "regla": "y_pred = 1 si score_attack >= umbral; score_attack = predict_proba[:, 1]",
        "denominadores_cero": "si TP+FN, TP+FP o FP+TN es 0, la métrica correspondiente se reporta como 0",
        "matriz_confusion": {"filas": "real", "columnas": "predicho", "orden": ["normal (0)", "ataque (1)"],
                             "valores": [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]},
        "metricas": {k: m[k] for k in ("recall", "precision", "fpr", "macro_f1", "accuracy")}
                    | ranking_metrics(y, s),
        "subconjunto_sin_entradas_vistas_en_train": {
            "filas": int((~seen).sum()),
            **{k: v for k, v in metrics_at(y[~seen], s[~seen], thr).items()
               if k in ("recall", "precision", "fpr", "macro_f1")}},
        "evaluacion": "Única evaluación de la solución congelada (2026-10-14, flow run Prefect 'tangible-tody'). "
                      "Estas métricas se recalculan desde predicciones-test.csv y coinciden con las originales.",
    }
    original = json.loads((ROOT / "reports/test_metrics.json").read_text(encoding="utf-8"))["test"]
    metrics["coincide_con_evaluacion_original"] = all(
        abs(metrics["metricas"][k] - original[k]) < 1e-12 for k in ("recall", "precision", "fpr", "macro_f1"))
    jdump(metrics, EV / "metricas-test.json")
    return metrics


def build_selection(versions: dict, exps: pd.DataFrame, test_metrics: dict):
    hf_file = ROOT / "configs" / "hf_revisions.json"
    hf = json.loads(hf_file.read_text()) if hf_file.exists() else {}
    def ref(v):  # noqa: E301
        cfg = json.loads((ROOT / f"models/{v}/config.json").read_text(encoding="utf-8"))
        return {"experimento": cfg["experiment"], "run_id": cfg["lineage"]["mlflow_run_id"],
                "umbral": cfg["threshold"], "model_sha256": cfg["model_sha256"],
                "git_commit_entrenamiento": cfg["lineage"]["git_commit_training"],
                "git_tag": v, "dvc": {"data/raw": cfg["lineage"]["dvc_data_raw_md5"],
                                      "data/processed": cfg["lineage"]["dvc_data_processed_md5"],
                                      f"models/{v}": dvc_md5(ROOT / f"models/{v}.dvc")},
                "huggingface": hf.get(v, "pendiente de publicar"),
                "validacion": {k: cfg["validation_metrics"][k] for k in ("recall", "precision", "fpr", "macro_f1")}}
    sel = {
        "version_elegida": "v2",
        **ref("v2"),
        "criterio": "Máximo recall de ataque en validación con FPR <= 5 %; desempate macro-F1. Escrito en "
                    "params.yaml (alert.max_fpr) antes de entrenar, commit 86d0212 (2026-10-14 22:52:53 UTC).",
        "razon": "E6 obtiene el mayor recall en validación (95,2 %) con FPR 5,0 %; E3 mostró que el cambio de familia "
                 "aporta +4,2 puntos frente a la regresión logística (v1) y E6 añade +0,5. El costo es de segundos.",
        "v1": ref("v1"),
        "comparacion_v1_v2_validacion": {
            "mejora": "recall 90,5 % → 95,2 % (+4,7 puntos) y macro-F1 0,911 → 0,945 con la misma FPR (5 %)",
            "empeora": "modelo menos interpretable que la logística (sin coeficientes directos) y artefacto más "
                       "pesado; la FPR no cambia (5 %) porque ambos umbrales se fijan con la misma política",
            "errores_aceptados": "~5 % de conexiones normales alarmadas y ~4,8 % de ataques no detectados en validación",
        },
        "congelacion_previa_al_test": {
            "params_yaml_commit": "86d0212 (2026-10-14 22:52:53 UTC) — criterio y experimentos antes de entrenar",
            "prefect": prefect_freeze_evidence(),
            "mlflow": "en el run b1caa191 la métrica val_recall se registró 22:58:26 UTC y test_recall 22:58:34 UTC",
            "git_commit_resultados": "548227a (2026-10-14) — reports/seleccion.json y reports/test_metrics.json",
            "nota": "La primera guía (2026-10-02) no pedía v1/v2; la elección de E6 se hizo con validación y no se "
                    "modificó al recibir la guía actualizada (2026-10-16). v1 no se usó para elegir y su "
                    "diagnóstico en test (reports/analisis_post_hoc.csv) es posterior y declarado como tal.",
        },
        "resultado_test": {k: test_metrics["metricas"][k] for k in ("recall", "precision", "fpr", "macro_f1")},
    }
    jdump(sel, EV / "seleccion.json")


def build_recovery_inputs(params: dict):
    rec = EV / "recuperacion"
    rec.mkdir(exist_ok=True)
    train = read_raw(RAW / TRAIN_CSV)
    _, va_idx = split_indices(train, params)
    val = train.iloc[va_idx]
    # 20 filas fijas de validación: 10 normales y 10 ataques, semilla 42
    sample = pd.concat([val[val[TARGET] == 0].sample(10, random_state=42),
                        val[val[TARGET] == 1].sample(10, random_state=42)]).sort_values(ID)
    sample.to_csv(rec / "entrada_validacion_20.csv", index=False)
    hashes = {"entrada_validacion_20.csv": sha256(rec / "entrada_validacion_20.csv")}
    for v in ("v1", "v2"):
        model, thr, _ = load(str(ROOT / f"models/{v}"))
        out = predict_frame(model, thr, sample, TRAIN_CSV)
        out.to_csv(rec / f"salida_original_{v}.csv", index=False, float_format="%.17g")
        cfg = json.loads((ROOT / f"models/{v}/config.json").read_text(encoding="utf-8"))
        hashes[f"salida_original_{v}.csv"] = sha256(rec / f"salida_original_{v}.csv")
        hashes[f"{v}_model_sha256"] = cfg["model_sha256"]
    jdump(hashes, rec / "hashes_originales.json")


def main():
    params = load_params()
    setup_mlflow(params)
    EV.mkdir(exist_ok=True)
    client = MlflowClient()
    versions = load_versions()
    test = build_audit(params)
    runs = runs_by_commit(client, params)
    export_mlflow(client, runs)
    exps = build_experiments(runs, params, versions)
    test_metrics = build_test(test, versions, exps)
    build_selection(versions, exps, test_metrics)
    build_recovery_inputs(params)
    print("Evidencias en", EV.relative_to(ROOT))
    print(exps[["id", "familia", "run_id", "umbral", "recall", "precision", "fpr", "macro_f1"]].round(4).to_string())
    print("Test v2:", {k: round(v, 4) for k, v in test_metrics["metricas"].items()},
          "| coincide con la evaluación original:", test_metrics["coincide_con_evaluacion_original"])


if __name__ == "__main__":
    main()
