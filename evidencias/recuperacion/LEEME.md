# Prueba de recuperación

Ejecutada el 2026-10-23 en una carpeta nueva, con un entorno virtual nuevo y sin reutilizar cachés
(pip con `PIP_CACHE_DIR` vacío y `--no-cache-dir`, caché DVC nueva dentro del clon, `HF_HOME` nuevo).
Mismo computador que el desarrollo (permitido por la guía). Duración total: 166 s.

## Comando ejecutado

```bash
PIP_CACHE_DIR=<CARPETA_NUEVA>/nocache bash scripts/recuperacion.sh <CARPETA_NUEVA>/defiende-el-servidor main
```

El script hace, en orden (ver [scripts/recuperacion.sh](../../scripts/recuperacion.sh)):

1. `git clone https://github.com/mjcastano29-tech/defiende-el-servidor.git` y `git checkout main`
   (commit `bf3bbb572abfc993217c636e9eff8c937d524fa0`).
2. `python3 -m venv .venv` (o venv sin pip + get-pip si el sistema no trae `python3-venv`) y
   `pip install --no-cache-dir -r requirements.txt`.
3. `dvc pull data/raw.dvc data/processed.dvc` desde el remoto público de Hugging Face;
   `sha256sum -c ../SHA256SUMS` y comparación de las huellas de los parquet con `auditoria.json`.
4. Descarga de v1 (`18ca32ce72d39f02782a38d745185500aa5625cb`) y v2 (`cab4044d0e3ae66a06ed1227bd4f5384e7808271`)
   por SHA completo, carga de preprocesamiento + modelo + umbral y predicción sobre las 20 filas fijas.
5. Comparación con las salidas guardadas en el entorno original.

## Archivos

| Archivo | Contenido |
|---|---|
| `entrada_validacion_20.csv` | 20 filas fijas de validación (10 normales, 10 ataques; `sample(random_state=42)`), con `id` y `label` |
| `salida_original_v1.csv`, `salida_original_v2.csv` | predicciones en el entorno original (`row_id, y_true, score_attack, y_pred`) |
| `hashes_originales.json` | SHA-256 de la entrada, de las salidas originales y de cada `model.skops` |
| `salida_recuperada_v1.csv`, `salida_recuperada_v2.csv` | predicciones en el entorno nuevo con los modelos descargados de Hugging Face |
| `comparacion.json` | coincidencia de `row_id` y `y_pred`, diferencia máxima de `score_attack`, hash del modelo descargado |
| `log_recuperacion.txt` | salida completa del script (la ruta de la carpeta temporal se reemplazó por `<CARPETA_NUEVA>`) |

## Resultado

| Versión | Revisión Hugging Face | `y_pred` iguales | máx. \|Δ score_attack\| | Hash del modelo | Resultado |
|---|---|---|---|---|---|
| v1 | `18ca32ce…` | sí (20/20) | 2,2e-16 | coincide | OK |
| v2 | `cab4044d…` | sí (20/20) | 1,1e-16 | coincide | OK |

Tolerancia declarada: 1e-9 en `score_attack` (las diferencias observadas son del orden del épsilon de
punto flotante de 64 bits).
