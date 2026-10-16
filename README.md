# Defiende el servidor

Detección de ataques de red con machine learning sobre **UNSW-NB15**, con trazabilidad
completa: Git (código y decisiones), DVC (datos y modelo), MLflow (experimentos y versiones
del modelo) y Prefect (orquestación del pipeline).

**Resultado:** gradient boosting (E6) con umbral 0,667 — en test detecta **96,9 %** de los ataques
con **18,2 %** de falsas alarmas (5 % en validación); ver [decision_final.md](reports/decision_final.md).

**Objetivo:** distinguir tráfico normal (`label=0`) de ataque (`label=1`) y elegir un umbral
de alerta defendible.

| Documento | Contenido |
|---|---|
| [reports/EDA.md](reports/EDA.md) | Estructura y contexto de los datos, hallazgos, decisiones de preparación |
| [reports/experimentos.md](reports/experimentos.md) | Hipótesis, cambio, resultado y conclusión de cada experimento |
| [reports/decision_final.md](reports/decision_final.md) | Modelo y umbral elegidos, errores, resultado del test, limitaciones |
| [reports/analisis_post_hoc.csv](reports/analisis_post_hoc.csv) | Por qué la FPR de validación no se mantuvo en test (`scripts/analisis_post_hoc.py`) |
| [data/README.md](data/README.md) | Fuente, versión, checksums y condiciones de uso de los datos |
| [imagenes_eda/](imagenes_eda/) · [imagenes_resultados/](imagenes_resultados/) | Figuras listas para ver |

## Estructura

```
params.yaml            configuración única: semilla, partición, política de alerta, experimentos + hipótesis
src/data.py            lectura, validación y partición (train/val agrupada por filas idénticas)
src/features.py        preprocesamiento (parte del modelo guardado)
src/train.py           entrena un experimento y lo registra en MLflow
src/evaluate.py        compara en validación, congela la elección, evalúa el test una vez, registra
src/predict.py         carga una versión guardada y predice
flows/pipeline.py      flujo de Prefect que conecta todas las etapas
notebooks/01_eda.ipynb análisis exploratorio
scripts/               descarga de datos y figuras del EDA
data/raw.dvc           ┐
data/processed.dvc     │ referencias DVC a datos, modelo y registros de MLflow
models/final.dvc       │ (el contenido vive en el remoto DVC)
mlflow.dvc             ┘
```

## Reproducir desde cero

Requisitos: Python 3.10+, Git. En Ubuntu/Debian, `sudo apt install python3-venv`.

```bash
git clone https://github.com/mjcastano29-tech/defiende-el-servidor.git
cd defiende-el-servidor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Opción A: recuperar exactamente los mismos datos y modelo (DVC)

```bash
dvc pull            # data/raw, data/processed, models/final y mlflow/
```

MLflow guarda rutas absolutas; si el repo queda en otra carpeta, `setup_mlflow` las reubica
automáticamente al primer uso. El remoto configurado es una carpeta local (`/home/mjcas/dvc-storage/defiende-el-servidor`).
En otra máquina, copia esa carpeta y apunta el remoto a ella:
`dvc remote modify local url /ruta/a/la/copia`.

### Opción B: reconstruir todo desde la fuente

```bash
scripts/download_data.sh            # CSV de train/test + verificación de checksums
python -m flows.pipeline            # prepara, entrena 7 experimentos, elige, evalúa test, registra, versiona
```

`scripts/download_data.sh --full` descarga además los 4 CSV completos (2,5 M de filas) del
SharePoint oficial de UNSW.

## Demostración

```bash
# 1. Ejecutar el pipeline y observar sus estados en Prefect
prefect server start                # en otra terminal; UI en http://127.0.0.1:4200
python -m flows.pipeline

# 2. Ver los experimentos y el modelo registrado en MLflow
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db    # http://127.0.0.1:5000

# 3. Cargar una versión guardada y predecir
python -m src.predict --input data/raw/UNSW_NB15_testing-set.csv --output reports/predicciones.csv
python -m src.predict --model "models:/defiende-el-servidor@champion" \
       --input data/raw/UNSW_NB15_testing-set.csv --output reports/predicciones.csv

# 4. Recuperar datos y modelo con DVC
rm -rf data/processed models/final && dvc pull && dvc status
# Volver a una versión anterior: git checkout <commit> -- '*.dvc' && dvc checkout
```

## Política del test

La elección de modelo y umbral se hace **solo con validación** y se congela en
`reports/seleccion.json`. El test se evalúa una vez con esa elección
(`reports/test_metrics.json`). Si una nueva ejecución eligiera otra cosa después de
haber visto el test, el pipeline falla a propósito; `--force-test` lo permite, pero debe
documentarse.

## Datos y atribución

UNSW-NB15 — Moustafa, N. & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network
intrusion detection systems*. MilCIS. Fuente: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>.
Uso académico. Es un benchmark sintético de laboratorio: los resultados no prueban
eficacia en una red actual.
