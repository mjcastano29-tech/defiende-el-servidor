# Entrega · MLOps — Defiende el servidor — Grupo A

## Integrantes y repositorios

- **Jorge Horacio Rojas Criollo** — Grupo A
- **Maria Jimena Castaño Albarracin** — Grupo A
- Código (GitHub): https://github.com/mjcastano29-tech/defiende-el-servidor
- **Commit final de la entrega:** tag Git `entrega` (rama `main`); el SHA completo va en el mensaje de entrega
- Modelos (Hugging Face Hub): https://huggingface.co/Marimar2006/defiende-el-servidor
- Almacenamiento DVC (Hugging Face, dataset público): https://huggingface.co/datasets/Marimar2006/defiende-el-servidor-dvc

## Versiones

| | **v1** | **v2** |
|---|---|---|
| Solución | E1 · regresión logística | E6 · HistGradientBoosting (500 it., 63 hojas, lr 0,05) |
| Commit de código que la entrenó | `80cdc9925e7af1d1c1aef0696b019995923ab0c5` | `80cdc9925e7af1d1c1aef0696b019995923ab0c5` |
| Commit/tag Git que la identifica | tag `v1` → `57c2c870a1d1dc172d5e7a5c93a2330aea1e56c4` | tag `v2` → `ea3a50fd3271f3731b6f63d64f70178d7de09ac9` |
| Datos DVC originales (`data/raw.dvc`) | md5 `ec34e6c98cfc4117e0b89a111496a0cb.dir` ¹ | md5 `ec34e6c98cfc4117e0b89a111496a0cb.dir` ¹ |
| Datos DVC preparados (`data/processed.dvc`) | md5 `f6c1bfa06596c6aaafc11b20e90df2e3.dir` | md5 `f6c1bfa06596c6aaafc11b20e90df2e3.dir` |
| Paquete del modelo en DVC | `models/v1.dvc` md5 `4e255fd0dd98e82128f88f108551b901.dir` | `models/v2.dvc` md5 `59085e5e8a446af42ae5d677ea1d7826.dir` |
| Run ID de MLflow | `ad45444b57f342e7b7df102aa9898abc` | `b1caa19139d54c2ea08d8984bcbb0399` |
| Hugging Face (SHA completo) | `18ca32ce72d39f02782a38d745185500aa5625cb` (tag `v1`) | `cab4044d0e3ae66a06ed1227bd4f5384e7808271` (tag `v2`) |
| SHA-256 de `model/model.skops` | `7cdc9965a67ccd0342e8b54ebddaf6b84121e3498ddea9dbde27a859f0e221bc` | `24b3db2a677a40333f04ae24a3368dabea72ed7f963954301f1c749f42132673` |
| Umbral (`y_pred=1` si `score_attack >= umbral`) | `0.788666967181885` | `0.6673794060957008` |
| Validación: recall / precisión / FPR / macro-F1 | 0,9049 / 0,9746 / 0,0500 / 0,9108 | 0,9522 / 0,9759 / 0,0499 / 0,9452 |

¹ Los runs se entrenaron con `data/raw` md5 `21cdb14982ac36a2bce75cc96cf51d4f.dir` (commit `86d0212`), que
además contenía los 4 CSV completos de UNSW-NB15. Después se dejó solo lo que pide el taller; los dos CSV
usados tienen el mismo SHA-256 en ambas versiones (ver `evidencias/auditoria.json`).

Revisión adicional de Hugging Face `eecb7a33a32e757350e6c19e60499c917ba786d0` (tag `v2-test-card`): añade el
resultado de test a la model card de v2; los archivos del modelo son idénticos a la revisión `cab4044d…`.

## Evidencias

| Pide la guía | Archivo |
|---|---|
| Auditoría de datos y particiones | [evidencias/auditoria.json](evidencias/auditoria.json) · asignación por fila en [evidencias/asignacion_splits.csv](evidencias/asignacion_splits.csv) |
| Tabla de experimentos | [evidencias/experimentos.csv](evidencias/experimentos.csv) |
| Selección congelada | [evidencias/seleccion.json](evidencias/seleccion.json) |
| Congelación anterior al test (registros fechados) | [evidencias/congelacion/](evidencias/congelacion/) (línea de tiempo Git/MLflow/Prefect) |
| Métricas de test | [evidencias/metricas-test.json](evidencias/metricas-test.json) |
| Predicciones de test (82.332 filas) | [evidencias/predicciones-test.csv](evidencias/predicciones-test.csv) |
| Registros de MLflow exportados | [evidencias/mlflow/](evidencias/mlflow/) (`runs.csv`, `runs/<run_id>.json`, `models/`, `LEEME.md`) |
| Prueba de recuperación | [evidencias/recuperacion/](evidencias/recuperacion/) (entrada, salidas originales y recuperadas, hashes, `comparacion.json`, log) |
| Configuración | [params.yaml](params.yaml) (equivale a `configs/`), [configs/versions.yaml](configs/versions.yaml), [configs/hf_revisions.json](configs/hf_revisions.json) |
| Informe | `informe.pdf` (adjunto) · fuentes en [reports/](reports/) |

## Acceso al almacenamiento DVC

> **Desde el ZIP de la entrega** (sin carpeta `.git`): ejecutar `git init` antes de `dvc pull`, porque DVC
> requiere un repositorio Git. Para ver el historial y los tags `v1`/`v2`, clonar desde GitHub.

El remoto por defecto (`.dvc/config`) es HTTP de solo lectura sobre un dataset público de Hugging Face; no
requiere cuenta ni token. `dvc pull` recupera datos originales, preparados, paquetes v1/v2 y la base de MLflow.

## Comandos

```bash
# Instalación
git clone https://github.com/mjcastano29-tech/defiende-el-servidor.git && cd defiende-el-servidor
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# Recuperación de datos
dvc pull && (cd data/raw && sha256sum -c ../SHA256SUMS)
# Entrenamiento de un experimento
python -m src.train --experiment E1_logreg
# Predicción con cada versión descargada por SHA
python -m src.predict --model hf://Marimar2006/defiende-el-servidor@18ca32ce72d39f02782a38d745185500aa5625cb --input evidencias/recuperacion/entrada_validacion_20.csv --output pred_v1.csv
python -m src.predict --model hf://Marimar2006/defiende-el-servidor@cab4044d0e3ae66a06ed1227bd4f5384e7808271 --input evidencias/recuperacion/entrada_validacion_20.csv --output pred_v2.csv
# Prueba de recuperación completa en carpeta y entorno nuevos
scripts/recuperacion.sh /ruta/nueva/defiende-el-servidor main
```

## Tiempo y recursos medidos

Equipo: Intel Core i5-1235U (12 hilos), 3,7 GB de RAM asignados a WSL2 (Ubuntu 22.04), sin GPU, Python 3.10.12.

| Tarea | Tiempo |
|---|---|
| Preparación de datos (`prepare`) | ~20 s |
| Entrenamiento E0–E6 | 0,6 s · 24,1 s · 16,6 s · 3,8 s · 3,9 s · 1,8 s · 14,5 s |
| Pipeline completo (7 experimentos + selección + test) | ~2 min 45 s |
| `dvc pull` desde el remoto público (clon sin caché) | ~22 s |
| Prueba de recuperación completa (incluye instalar dependencias) | 166 s (clon, venv, instalación sin caché, dvc pull, descarga y predicción v1/v2) |
| Memoria máxima al generar evidencias | ~0,8 GB |

## Limitaciones conocidas

- UNSW-NB15 es tráfico sintético de laboratorio (2015): los resultados no demuestran eficacia en una red actual.
- Prevalencia de ataque irreal (68 % entrenamiento, 55 % test); en una red real la precisión sería menor.
- La FPR fijada en validación (5 %) sube a 18,2 % en test: el tráfico normal del archivo de test es distinto
  (41 % con `sttl=254` frente a 20 %). El umbral debe recalibrarse con tráfico normal de la red de destino.
- 10,37 % de las filas de test tienen entradas idénticas a filas de entrenamiento/validación; sin ellas, el
  resultado apenas cambia (recall 0,9634, FPR 0,1773).
- **Integridad del test.** El test se evaluó una sola vez, el 2026-10-14, cuando regía la primera guía. El
  criterio estaba en Git antes de entrenar (`86d0212`, 22:52:53 UTC) y el pipeline congeló la elección antes de
  tocar el test (Prefect: `comparar-y-congelar` termina 22:58:31.803, `evaluar-test` empieza 22:58:31.808; MLflow:
  última métrica de validación 22:58:26, primera de test 22:58:34). La guía actualizada (2026-10-16) no cambió
  la decisión: v2 es exactamente esa elección. No existe un commit de `seleccion.json` anterior al test porque
  elección y evaluación ocurrieron en la misma ejecución; los registros fechados están exportados en
  `evidencias/congelacion/`.
- El diagnóstico post-hoc de E1/E3/E4 en test (`reports/analisis_post_hoc.csv`) es posterior a la selección
  y no la modificó.

## Uso de IA, código externo y contribuciones

- **IA:** se usó Claude Code (Anthropic) como asistente para: descarga y verificación de los datos, EDA,
  código del pipeline (preparación, entrenamiento, evaluación, empaquetado, predicción), publicación en
  Hugging Face, scripts de evidencias y recuperación, y borradores de documentación. El equipo revisó el
  código y los resultados; se verificaron las cifras recalculándolas desde los archivos de evidencia y con
  la prueba de recuperación en un entorno nuevo.
- **Código externo:** no se copió código de terceros; se usan las bibliotecas de `requirements.txt` con sus
  licencias (scikit-learn BSD-3, pandas BSD-3, MLflow Apache-2.0, DVC Apache-2.0, huggingface_hub Apache-2.0,
  Prefect Apache-2.0). El mirror de datos `Nir-J/ML-Projects` se usó solo como fuente de los CSV, verificada
  por SHA-256.
- **Contribuciones:** trabajo conjunto. Jorge Horacio Rojas Criollo y Maria Jimena Castaño Albarracin
  participaron ambos en todas las etapas —auditoría y preparación de datos, diseño de hipótesis y
  experimentos, elección de umbral, versionado con Git/DVC/MLflow/Hugging Face, prueba de recuperación e
  informe—, revisando y validando juntos el trabajo asistido por IA. Ambos pueden explicar las decisiones
  del proyecto.
- **Equipo de trabajo:** desarrollamos el taller juntos en clase desde el computador de Maria Jimena. Por eso
  todos los commits aparecen con su usuario de GitHub (`mjcastano29-tech`) y los repositorios de Hugging Face
  están en su cuenta (`Marimar2006`); la autoría en las plataformas no refleja el reparto del trabajo, que fue
  conjunto.
