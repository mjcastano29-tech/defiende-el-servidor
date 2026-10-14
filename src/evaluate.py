"""Comparación en validación, evaluación única en test, registro y exportación del modelo."""
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow import MlflowClient

from src.config import MODEL_DIR, REPORTS, RESULT_FIGURES, ROOT, git_commit
from src.data import FAMILY, TARGET, load_split
from src.metrics import metrics_at, ranking_metrics
from src.features import TRUSTED_TYPES
from src.train import predict_scores, setup_mlflow

SELECTION_FILE = REPORTS / "seleccion.json"
TEST_FILE = REPORTS / "test_metrics.json"
C1, C2, GRAY, INK2 = "#2a78d6", "#eb6834", "#8a8984", "#52514e"


def compare(results: list[dict], params: dict) -> dict:
    """Elige modelo y umbral SOLO con validación: mayor recall con FPR <= max_fpr; desempate macro-F1."""
    table = pd.DataFrame(results)
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / "validacion.csv", index=False)
    ranked = table.sort_values(["val_recall", "val_macro_f1"], ascending=False)
    best = ranked.iloc[0]
    selection = {
        "experiment": best["name"], "run_id": best["run_id"], "threshold": float(best["threshold"]),
        "criterion": f"max recall en validación con FPR <= {params['alert']['max_fpr']}; desempate macro-F1",
        "val_recall": float(best["val_recall"]), "val_fpr": float(best["val_fpr"]),
        "val_precision": float(best["val_precision"]), "val_macro_f1": float(best["val_macro_f1"]),
    }
    _plot_validation(table, params)
    return selection


def freeze(selection: dict, force: bool = False) -> None:
    """Congela la elección. Si ya se evaluó el test con OTRA elección, falla: eso sería elegir mirando el test."""
    if TEST_FILE.exists() and not force:
        previous = json.loads(TEST_FILE.read_text(encoding="utf-8"))["selection"]
        same = (previous["experiment"] == selection["experiment"]
                and abs(previous["threshold"] - selection["threshold"]) < 1e-9)
        if not same:
            raise RuntimeError(
                f"El test ya se evaluó con {previous['experiment']} (umbral {previous['threshold']:.4f}) y ahora "
                f"la validación elige {selection['experiment']}. Cambiar la elección después de ver el test "
                "invalida la evaluación; si el cambio es legítimo (nuevos datos/experimentos), usa --force y documéntalo.")
    SELECTION_FILE.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_test(selection: dict, params: dict) -> dict:
    setup_mlflow(params)
    model = mlflow.sklearn.load_model(f"runs:/{selection['run_id']}/model")
    test = load_split(params, "test")
    score = predict_scores(model, test)
    y, thr = test[TARGET].to_numpy(), selection["threshold"]
    unseen = ~test["seen_in_train"].to_numpy()

    pred = score >= thr
    by_family = (test.assign(pred=pred).groupby(FAMILY)
                 .agg(filas=("pred", "size"), marcadas_como_ataque=("pred", "mean")).round(4))
    result = {
        "selection": selection,
        "test": {**metrics_at(y, score, thr), **ranking_metrics(y, score)},
        "test_sin_filas_vistas_en_train": {**metrics_at(y[unseen], score[unseen], thr),
                                          "rows": int(unseen.sum())},
        "test_umbral_0.5": metrics_at(y, score, 0.5),
        "por_familia": by_family.reset_index().to_dict(orient="records"),
        "git_commit": git_commit(),
    }
    TEST_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # El test se registra en la MISMA ejecución que produjo el modelo
    with mlflow.start_run(run_id=selection["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in result["test"].items()})
        mlflow.log_metrics({f"test_unseen_{k}": v for k, v in result["test_sin_filas_vistas_en_train"].items()})
        mlflow.set_tag("selected", "true")
        mlflow.log_artifact(str(TEST_FILE))
    _plot_test(result)
    return result


def register_and_export(selection: dict, params: dict) -> dict:
    """Registra la versión en el Model Registry (alias champion) y la exporta a models/final."""
    setup_mlflow(params)
    name = params["mlflow"]["registered_model"]
    mv = mlflow.register_model(f"runs:/{selection['run_id']}/model", name)
    client = MlflowClient()
    client.set_registered_model_alias(name, "champion", mv.version)
    for k in ("experiment", "threshold", "criterion"):
        client.set_model_version_tag(name, mv.version, k, str(selection[k]))

    model = mlflow.sklearn.load_model(f"runs:/{selection['run_id']}/model")
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(model, str(MODEL_DIR), skops_trusted_types=TRUSTED_TYPES)
    meta = {
        "registered_model": name, "version": int(mv.version), "alias": "champion",
        "run_id": selection["run_id"], "experiment": selection["experiment"],
        "threshold": selection["threshold"], "git_commit": git_commit(),
        "data_sha256": json.loads((ROOT / params["data"]["processed_dir"] / "split_stats.json")
                                  .read_text())["source_sha256"],
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _plot_validation(table: pd.DataFrame, params: dict) -> None:
    RESULT_FIGURES.mkdir(exist_ok=True)
    t = table.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, col, title, color in [(axes[0], "val_recall", "Ataques detectados (recall)", C2),
                                  (axes[1], "val_fpr", "Normales alarmadas (FPR)", C1)]:
        bars = ax.barh(t["name"], t[col] * 100, color=color, height=0.6)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, color=INK2, padding=3)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, 110)
        ax.set_xlabel("% (validación)")
    axes[1].axvline(params["alert"]["max_fpr"] * 100, color=GRAY, linestyle="--", linewidth=1)
    fig.suptitle(f"Comparación en validación con el umbral que respeta FPR ≤ {params['alert']['max_fpr']:.0%}",
                 x=0.01, ha="left", fontsize=13, weight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULT_FIGURES / "01_validacion.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_test(result: dict) -> None:
    RESULT_FIGURES.mkdir(exist_ok=True)
    m = result["test"]
    fam = pd.DataFrame(result["por_familia"]).sort_values("marcadas_como_ataque")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})

    ax = axes[0]
    cm = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    ax.imshow([[1, 0], [0, 1]], cmap=matplotlib.colors.ListedColormap(["#f3f2ee", "#dbe8f8"]))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], ["pred. normal", "pred. ataque"])
    ax.set_yticks([0, 1], ["real normal", "real ataque"])
    ax.set_title(f"Test · recall {m['recall']:.1%} · FPR {m['fpr']:.1%} · precisión {m['precision']:.1%}",
                 loc="left", fontsize=10)

    ax = axes[1]
    colors = [C1 if f == "Normal" else C2 for f in fam[FAMILY]]
    bars = ax.barh(fam[FAMILY], fam["marcadas_como_ataque"] * 100, color=colors, height=0.6)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=8, color=INK2, padding=3)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% marcado como ataque (test)")
    ax.set_title("Detección por familia (Normal = falsas alarmas)", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Evaluación final en test · {result['selection']['experiment']} · umbral "
                 f"{result['selection']['threshold']:.3f}", x=0.01, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(RESULT_FIGURES / "02_test.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
