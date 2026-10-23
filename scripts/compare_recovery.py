"""Paso de verificación de la prueba de recuperación (lo llama scripts/recuperacion.sh).

  data    verifica las huellas de los datos preparados contra evidencias/auditoria.json
  models  descarga v1/v2 de Hugging Face por SHA (configs/hf_revisions.json), predice sobre las 20 filas fijas
          de validación y compara con las salidas guardadas en el entorno original
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import ROOT  # noqa: E402
from src.package import sha256  # noqa: E402

REC = ROOT / "evidencias" / "recuperacion"
TOLERANCE = 1e-9  # diferencia absoluta máxima aceptada en score_attack


def check_data() -> None:
    audit = json.loads((ROOT / "evidencias" / "auditoria.json").read_text(encoding="utf-8"))
    ok = True
    for name, expected in audit["particiones"]["huellas_parquet_sha256"].items():
        got = sha256(ROOT / "data" / "processed" / f"{name}.parquet")
        ok &= got == expected
        print(f"data/processed/{name}.parquet  {got}  {'OK' if got == expected else 'NO COINCIDE'}")
    for f in audit["archivos"]:
        got = sha256(ROOT / "data" / "raw" / f["archivo"])
        ok &= got == f["sha256_anexo_taller"]
        print(f"data/raw/{f['archivo']}  {got}  {'OK (anexo del taller)' if got == f['sha256_anexo_taller'] else 'NO'}")
    if not ok:
        raise SystemExit("Las huellas de los datos recuperados no coinciden")


def check_models() -> None:
    from huggingface_hub import snapshot_download

    from src.data import read_raw
    from src.predict import load, predict_frame

    revisions = json.loads((ROOT / "configs" / "hf_revisions.json").read_text(encoding="utf-8"))
    entrada = REC / "entrada_validacion_20.csv"
    original_hashes = json.loads((REC / "hashes_originales.json").read_text(encoding="utf-8"))
    assert sha256(entrada) == original_hashes["entrada_validacion_20.csv"], "la entrada fija cambió"
    sample = read_raw(entrada)
    result = {"entrada": {"archivo": entrada.name, "filas": len(sample), "sha256": sha256(entrada)},
              "tolerancia_abs_score": TOLERANCE, "versiones": {}}
    for v in ("v1", "v2"):
        rev = revisions[v]
        folder = Path(snapshot_download(rev["repo"], revision=rev["sha"], token=False))
        model_sha = sha256(folder / "model" / "model.skops")
        model, thr, desc = load(str(folder))
        out = predict_frame(model, thr, sample, "UNSW_NB15_training-set.csv")
        out.to_csv(REC / f"salida_recuperada_{v}.csv", index=False, float_format="%.17g")
        orig = pd.read_csv(REC / f"salida_original_{v}.csv")
        diff = float(np.abs(out["score_attack"].to_numpy() - orig["score_attack"].to_numpy()).max())
        same_ids = bool((out["row_id"] == orig["row_id"]).all())
        same_pred = bool((out["y_pred"].to_numpy() == orig["y_pred"].to_numpy()).all())
        ok = same_ids and same_pred and diff <= TOLERANCE and model_sha == rev["model_sha256"]
        result["versiones"][v] = {
            "huggingface": f"{rev['repo']}@{rev['sha']}", "experimento": rev["experiment"],
            "umbral": thr, "model_sha256_descargado": model_sha, "model_sha256_esperado": rev["model_sha256"],
            "salida_recuperada_sha256": sha256(REC / f"salida_recuperada_{v}.csv"),
            "salida_original_sha256": original_hashes[f"salida_original_{v}.csv"],
            "row_id_coinciden": same_ids, "y_pred_coinciden": same_pred, "max_abs_diff_score": diff,
            "alertas": int(out["y_pred"].sum()), "resultado": "OK" if ok else "FALLA",
        }
        print(f"{v} ({desc}): model.skops {model_sha[:12]}… · y_pred iguales={same_pred} · "
              f"max|Δscore|={diff:.3g} · {'OK' if ok else 'FALLA'}")
    result["resultado"] = "OK" if all(r["resultado"] == "OK" for r in result["versiones"].values()) else "FALLA"
    (REC / "comparacion.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Resultado global:", result["resultado"])
    if result["resultado"] != "OK":
        raise SystemExit(1)


if __name__ == "__main__":
    {"data": check_data, "models": check_models}[sys.argv[1]]()
