# Defiende el servidor

Clasificación binaria de conexiones de red **normal (0) / ataque (1)** con UNSW-NB15, con código,
datos, experimentos y modelos versionados y recuperables.

**Integrantes:** Jorge Horacio Rojas Criollo y Maria Jimena Castaño Albarracin — MLOps 2026-II, Grupo A.
Índice de la entrega y evidencias: **[entrega.md](entrega.md)**.

| Pieza | Herramienta | Dónde |
|---|---|---|
| Código | Git/GitHub | este repositorio · tags `v1` y `v2` |
| Datos originales y preparados | DVC | `data/raw.dvc`, `data/processed.dvc` · remoto público [Marimar2006/defiende-el-servidor-dvc](https://huggingface.co/datasets/Marimar2006/defiende-el-servidor-dvc) |
| Experimentos | MLflow | `mlflow.dvc` (base + artefactos) · exportación en [evidencias/mlflow/](evidencias/mlflow/) |
| Modelos v1/v2 | Hugging Face Hub | [Marimar2006/defiende-el-servidor](https://huggingface.co/Marimar2006/defiende-el-servidor) · revisiones en [configs/hf_revisions.json](configs/hf_revisions.json) |

| Versión | Modelo | Umbral | Validación (recall / precisión / FPR / macro-F1) | Hugging Face (SHA) |
|---|---|---:|---|---|
| v1 | E1 · regresión logística | 0,788667 | 0,9049 / 0,9746 / 0,0500 / 0,9108 | `18ca32ce72d39f02782a38d745185500aa5625cb` |
| v2 | E6 · HistGradientBoosting | 0,667379 | 0,9522 / 0,9759 / 0,0499 / 0,9452 | `cab4044d0e3ae66a06ed1227bd4f5384e7808271` |

Test final (solo v2, evaluado una vez): recall 0,9688 · precisión 0,8673 · FPR 0,1816 · macro-F1 0,8984
([evidencias/metricas-test.json](evidencias/metricas-test.json)).

## Requisitos

- **Python 3.10** (probado con 3.10.12 en Ubuntu 22.04 / WSL2), Git, conexión a internet.
- Dependencias fijadas en [requirements.txt](requirements.txt) (directas en `requirements.in`).
- No se necesita GPU, cuenta de Hugging Face ni credenciales: todos los repositorios son públicos.

## Instalación

```bash
git clone https://github.com/mjcastano29-tech/defiende-el-servidor.git
cd defiende-el-servidor
python3 -m venv .venv          # en Debian/Ubuntu puede requerir: sudo apt install python3-venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Recuperar los datos (DVC)

```bash
dvc pull data/raw.dvc data/processed.dvc       # originales y preparados desde el remoto público
(cd data/raw && sha256sum -c ../SHA256SUMS)    # huellas; coinciden con el anexo del taller
```

`dvc pull` sin argumentos recupera además `models/v1`, `models/v2` y `mlflow/` (registros de MLflow).
Alternativa sin DVC para los dos CSV: `scripts/download_data.sh`.

## Ejecutar un experimento

```bash
python -m src.train --experiment E1_logreg     # o E0_referencia … E6_hgb_complejo (params.yaml)
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db   # http://127.0.0.1:5000 (tras `dvc pull mlflow.dvc`)
```

Cada experimento usa las particiones fijas de `data/processed`, elige su umbral en validación
(máximo recall con FPR ≤ 5 %) y registra en MLflow hipótesis, parámetros, semilla, commit, datasets,
métricas y el modelo completo. Para regenerar las particiones: `python -c "from src.config import load_params; from src.data import prepare; prepare(load_params())"`.
El pipeline completo (opcional, Prefect) es `python -m flows.pipeline --skip-dvc`.

## Descargar cada modelo y predecir

```bash
# v1 y v2 desde Hugging Face, por SHA completo
python -m src.predict --model hf://Marimar2006/defiende-el-servidor@18ca32ce72d39f02782a38d745185500aa5625cb \
       --input evidencias/recuperacion/entrada_validacion_20.csv --output pred_v1.csv
python -m src.predict --model hf://Marimar2006/defiende-el-servidor@cab4044d0e3ae66a06ed1227bd4f5384e7808271 \
       --input evidencias/recuperacion/entrada_validacion_20.csv --output pred_v2.csv

# o desde las copias versionadas con DVC
python -m src.predict --model models/v2 --input data/raw/UNSW_NB15_testing-set.csv --output pred_test.csv
```

Salida: `row_id, y_true, score_attack, y_pred`, con `y_pred = 1` si `score_attack >= umbral` de la versión.

## Prueba de recuperación

```bash
scripts/recuperacion.sh /ruta/nueva/defiende-el-servidor main
```

Clona el código, crea un entorno nuevo, recupera datos con DVC sin caché previa, descarga v1 y v2 por
SHA y compara sus predicciones sobre 20 filas fijas de validación con las guardadas. Resultado y log en
[evidencias/recuperacion/](evidencias/recuperacion/).

## Estructura

```
params.yaml               semilla, partición, política de alerta y 7 experimentos con hipótesis
configs/versions.yaml     definición de v1/v2 (run de MLflow de cada una)
configs/hf_revisions.json revisiones de Hugging Face (SHA completo) y hashes del modelo
src/data.py               lectura, validación y partición agrupada por entradas idénticas
src/features.py           preprocesamiento (parte del modelo guardado)
src/train.py              entrena un experimento y lo registra en MLflow
src/evaluate.py           comparación en validación, congelación y evaluación única del test
src/package.py            empaqueta v1/v2 (modelo + config + model card) en models/<versión>
src/predict.py            carga una versión (carpeta o hf://repo@SHA) y predice
scripts/                  evidencias, publicación en Hugging Face, recuperación, figuras
flows/pipeline.py         orquestación opcional con Prefect
evidencias/               auditoría, experimentos, selección, test, MLflow y recuperación
reports/                  EDA, comparación de experimentos y decisión final
```

## Datos y atribución

UNSW-NB15 — Moustafa, N. & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion
detection systems*. MilCIS, IEEE. <https://research.unsw.edu.au/projects/unsw-nb15-dataset>. Uso académico.
Es un benchmark histórico de laboratorio: los resultados no demuestran eficacia sobre una red actual.
