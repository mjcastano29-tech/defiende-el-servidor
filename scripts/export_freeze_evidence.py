"""Exporta los registros fechados que muestran que la selección se congeló antes de evaluar el test.

Fuentes:
  - Git: commits del criterio (params.yaml) y de los resultados
  - MLflow (mlflow/mlflow.db, recuperable con `dvc pull mlflow.dvc`): historial de métricas con timestamp
    del run elegido (validación antes que test)
  - Prefect (base local ~/.prefect/prefect.db del equipo donde se ejecutó): estados y horas de cada tarea
    del flujo que hizo la primera evaluación de test

Escribe evidencias/congelacion/{linea_de_tiempo.json, mlflow_metricas_run_v2.csv, prefect_tareas.csv, LEEME.md}.
Uso: python scripts/export_freeze_evidence.py
"""
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from src.config import ROOT, load_params  # noqa: E402
from src.train import setup_mlflow  # noqa: E402

OUT = ROOT / "evidencias" / "congelacion"
RUN_V2 = "b1caa19139d54c2ea08d8984bcbb0399"
FLOW_RUN = "tangible-tody"
COMMITS = {"86d0212": "criterio de selección e hipótesis en params.yaml (antes de entrenar)",
           "80cdc99": "código con el que se ejecutó el pipeline (runs canónicos)",
           "548227a": "commit de resultados: seleccion.json y test_metrics.json"}


def norm(t) -> pd.Timestamp:
    """Hora en UTC con zona explícita (Prefect guarda UTC sin zona; Git la guarda con desfase)."""
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def utc_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="milliseconds")


def git_commits() -> list[dict]:
    rows = []
    for sha, desc in COMMITS.items():
        full, date = subprocess.run(["git", "show", "-s", "--format=%H|%cI", sha], cwd=ROOT, capture_output=True,
                                    text=True, check=True).stdout.strip().split("|")
        rows.append({"fuente": "git", "evento": desc, "referencia": full,
                     "hora_utc": datetime.fromisoformat(date).astimezone(timezone.utc).isoformat()})
    return rows


def mlflow_history() -> pd.DataFrame:
    setup_mlflow(load_params())
    client = MlflowClient()
    run = client.get_run(RUN_V2)
    rows = []
    for key in sorted(run.data.metrics):
        for m in client.get_metric_history(RUN_V2, key):
            rows.append({"run_id": RUN_V2, "metrica": key, "valor": m.value, "timestamp_ms": m.timestamp,
                         "hora_utc": utc_ms(m.timestamp),
                         "tipo": "test" if key.startswith("test_") else "validacion/entrenamiento"})
    df = pd.DataFrame(rows).sort_values(["timestamp_ms", "metrica"])
    df.to_csv(OUT / "mlflow_metricas_run_v2.csv", index=False)
    return df


def prefect_tasks() -> pd.DataFrame:
    db = Path.home() / ".prefect" / "prefect.db"
    con = sqlite3.connect(db)
    fr_id, fr_state, fr_start, fr_end = con.execute(
        "SELECT id, state_name, start_time, end_time FROM flow_run WHERE name = ?", (FLOW_RUN,)).fetchone()
    df = pd.read_sql_query(
        "SELECT name AS tarea, state_name AS estado, start_time AS inicio_utc, end_time AS fin_utc "
        "FROM task_run WHERE flow_run_id = ? ORDER BY start_time", con, params=(fr_id,))
    df.insert(0, "flow_run", FLOW_RUN)
    df.insert(1, "flow_run_id", fr_id)
    df.to_csv(OUT / "prefect_tareas.csv", index=False)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commits = git_commits()
    hist = mlflow_history()
    tasks = prefect_tasks()
    last_val = hist[hist.tipo != "test"]["timestamp_ms"].max()
    first_test = hist[hist.tipo == "test"]["timestamp_ms"].min()
    freeze = tasks[tasks.tarea.str.startswith("comparar")].iloc[0]
    evaluate = tasks[tasks.tarea.str.startswith("evaluar-test")].iloc[0]
    events = commits + [
        {"fuente": "prefect", "evento": "inicio del flujo (entrenamiento de E0–E6)",
         "referencia": FLOW_RUN, "hora_utc": tasks.inicio_utc.min()},
        {"fuente": "mlflow", "evento": "última métrica de validación del run elegido",
         "referencia": RUN_V2, "hora_utc": utc_ms(last_val)},
        {"fuente": "prefect", "evento": "fin de comparar-y-congelar (elección escrita en seleccion.json)",
         "referencia": freeze.tarea, "hora_utc": freeze.fin_utc},
        {"fuente": "prefect", "evento": "inicio de evaluar-test", "referencia": evaluate.tarea,
         "hora_utc": evaluate.inicio_utc},
        {"fuente": "mlflow", "evento": "primera métrica de test del run elegido",
         "referencia": RUN_V2, "hora_utc": utc_ms(first_test)},
    ]
    for e in events:
        e["hora_utc"] = norm(e["hora_utc"]).isoformat(timespec="milliseconds")
    timeline = sorted(events, key=lambda e: norm(e["hora_utc"]))
    checks = {
        "criterio_en_git_antes_de_entrenar": norm(commits[0]["hora_utc"]) < norm(tasks.inicio_utc.min()),
        "validacion_registrada_antes_que_test_en_mlflow": bool(last_val < first_test),
        "congelacion_termina_antes_de_evaluar_test_en_prefect": norm(freeze.fin_utc) < norm(evaluate.inicio_utc),
    }
    (OUT / "linea_de_tiempo.json").write_text(json.dumps(
        {"run_v2": RUN_V2, "flow_run_prefect": FLOW_RUN, "eventos": timeline,
         "comprobaciones": {k: bool(v) for k, v in checks.items()}}, indent=2, ensure_ascii=False), encoding="utf-8")
    for e in timeline:
        print(f"{str(e['hora_utc'])[:23]}  {e['fuente']:8s} {e['evento']}")
    print(checks)


if __name__ == "__main__":
    main()
