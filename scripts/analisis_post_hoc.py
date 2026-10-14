"""Análisis POST-HOC (después de ver el test): ¿por qué la FPR de validación no se mantuvo en test?

No cambia la selección congelada en reports/seleccion.json. Escribe reports/analisis_post_hoc.csv.
Uso: python scripts/analisis_post_hoc.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import REPORTS, load_params  # noqa: E402
from src.data import load_split  # noqa: E402
from src.metrics import metrics_at  # noqa: E402
from src.train import predict_scores, setup_mlflow  # noqa: E402

MODELS = ["E1_logreg", "E3_hgb", "E4_hgb_sin_ttl", "E6_hgb_complejo"]


def main():
    params = load_params()
    setup_mlflow(params)
    runs = pd.read_csv(REPORTS / "validacion.csv").set_index("name")
    val, test = load_split(params, "val"), load_split(params, "test")
    test_ttl_mix = test.loc[test.label == 0, "sttl"].value_counts(normalize=True)

    rows = {}
    for name in MODELS:
        model = mlflow.sklearn.load_model(f"runs:/{runs.loc[name, 'run_id']}/model")
        thr = runs.loc[name, "threshold"]
        # (a) Solo validación: FPR esperada si los normales tuvieran la mezcla de sttl del test
        normal = val[val.label == 0].assign(fp=lambda d: predict_scores(model, d) >= thr)
        fpr_by_ttl = normal.groupby("sttl")["fp"].mean()
        common = fpr_by_ttl.index.intersection(test_ttl_mix.index)
        reweighted = (fpr_by_ttl[common] * test_ttl_mix[common]).sum() / test_ttl_mix[common].sum()
        # (b) Diagnóstico en test, declarado post-hoc
        m = metrics_at(test.label.to_numpy(), predict_scores(model, test), thr)
        rows[name] = {
            "val_recall": runs.loc[name, "val_recall"], "val_fpr": runs.loc[name, "val_fpr"],
            "val_fpr_mezcla_ttl_test": reweighted,
            **{f"posthoc_test_{k}": m[k] for k in ("recall", "fpr", "precision", "macro_f1")},
        }
    out = pd.DataFrame(rows).T.round(4)
    out.to_csv(REPORTS / "analisis_post_hoc.csv")
    print(out.to_string())


if __name__ == "__main__":
    main()
