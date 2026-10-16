"""Pipeline completo orquestado con Prefect.

  python -m flows.pipeline                  # todo: datos → experimentos → selección → test → registro → DVC
  python -m flows.pipeline --skip-dvc       # sin versionar al final
  python -m flows.pipeline --force-test     # permite re-evaluar el test con otra selección (documentarlo)

Cada etapa es una tarea de Prefect: si una falla, el flujo queda en estado Failed y el
log indica qué etapa y por qué.
"""
import argparse
import subprocess
import sys

from prefect import flow, get_run_logger, task

from src.config import ROOT, load_params
from src import data, evaluate, train


@task(name="preparar-datos")
def preparar(params: dict) -> dict:
    stats = data.prepare(params)
    get_run_logger().info("Particiones: %s · tasa de ataque: %s · test visto en train: %s",
                          stats["rows"], stats["attack_rate"], stats["test_seen_in_train"])
    return stats


@task(name="entrenar", task_run_name="entrenar-{spec[name]}")
def entrenar(spec: dict, params: dict) -> dict:
    r = train.run_experiment(spec, params)
    get_run_logger().info("%s · recall %.4f · FPR %.4f · macro-F1 %.4f · umbral %.4f · %ss",
                          r["name"], r["val_recall"], r["val_fpr"], r["val_macro_f1"], r["threshold"], r["fit_seconds"])
    return r


@task(name="comparar-y-congelar")
def comparar(results: list[dict], params: dict, force: bool) -> dict:
    selection = evaluate.compare(results, params)
    evaluate.freeze(selection, force=force)
    get_run_logger().info("Elegido por validación: %s con umbral %.4f", selection["experiment"], selection["threshold"])
    return selection


@task(name="evaluar-test")
def evaluar_test(selection: dict, params: dict) -> dict:
    result = evaluate.evaluate_test(selection, params)
    t = result["test"]
    get_run_logger().info("TEST · recall %.4f · FPR %.4f · precisión %.4f · macro-F1 %.4f",
                          t["recall"], t["fpr"], t["precision"], t["macro_f1"])
    return result


@task(name="registrar-modelo")
def registrar(selection: dict, params: dict) -> dict:
    meta = evaluate.register(selection, params)
    get_run_logger().info("Registrado %s v%s (alias champion)",
                          meta["registered_model"], meta["version"])
    return meta


@task(name="versionar-dvc")
def versionar_dvc() -> None:
    log = get_run_logger()
    for cmd in (["add", "data/processed", "mlflow"], ["push"]):
        p = subprocess.run([sys.executable, "-m", "dvc", *cmd], cwd=ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"dvc {' '.join(cmd)} falló:\n{p.stderr or p.stdout}")
        log.info("dvc %s OK", " ".join(cmd))
    log.info("Ahora: git add *.dvc && git commit (guarda la referencia a esta versión)")


@flow(name="defiende-el-servidor")
def pipeline(skip_dvc: bool = False, force_test: bool = False) -> dict:
    params = load_params()
    preparar(params)
    results = [entrenar(spec, params) for spec in params["experiments"]]  # en serie: poca RAM
    selection = comparar(results, params, force_test)
    test = evaluar_test(selection, params)
    meta = registrar(selection, params)
    if not skip_dvc:
        versionar_dvc()
    return {"selection": selection, "test": test["test"], "model": meta}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dvc", action="store_true")
    ap.add_argument("--force-test", action="store_true")
    a = ap.parse_args()
    pipeline(skip_dvc=a.skip_dvc, force_test=a.force_test)
