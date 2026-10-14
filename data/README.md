# Datos

## Fuente y versión

| | |
|---|---|
| Dataset | UNSW-NB15 (UNSW Canberra Cyber Range Lab, 2015) |
| Página oficial | <https://research.unsw.edu.au/projects/unsw-nb15-dataset> |
| Archivos oficiales | SharePoint de UNSW → `UNSW-NB15 dataset/CSV Files/`, última modificación **2024-02-09** |
| Copia usada para train/test | GitHub `Nir-J/ML-Projects`, commit `44bbd02b9e70f74dd915218d5eb702f125341ffc` (indicada por el docente) |
| Verificación | La copia de GitHub es **idéntica byte a byte** a la descarga oficial (mismo SHA-256) |
| Versión DVC | `data/raw.dvc` (md5 del directorio) |

## Archivos en `data/raw/`

| Archivo | Filas | Uso |
|---|---:|---|
| `UNSW_NB15_training-set.csv` | 175.341 | entrenamiento + validación |
| `UNSW_NB15_testing-set.csv` | 82.332 | evaluación final (una sola vez) |
| `UNSW-NB15_1..4.csv` | 2.540.044 (sin encabezado) | dataset completo, no usado en el modelo |
| `NUSW-NB15_GT.csv` | 188.913 | ground truth de eventos de ataque |
| `NUSW-NB15_features.csv` | 49 | descripción de variables |
| `UNSW-NB15_LIST_EVENTS.csv` | 208 | categorías y subcategorías de ataque |
| `The UNSW-NB15 description.pdf` | — | descripción oficial |

Checksums: [`SHA256SUMS`](SHA256SUMS) (`cd data/raw && sha256sum -c ../SHA256SUMS`).

Notas: los CSV de train/test traen BOM (leer con `encoding="utf-8-sig"`). El docente advierte
que en algunas copias los nombres train/test están intercambiados; el pipeline verifica el número
de filas y falla si no coincide.

## `data/processed/` (generado por el pipeline)

`train.parquet` (140.779) y `val.parquet` (34.562) salen del train oficial con
`StratifiedGroupKFold` (5 pliegues, semilla 42, se usa el primero), agrupando filas con
variables idénticas para que ninguna copia quede a ambos lados. `test.parquet` es el test
oficial con la columna `seen_in_train` (8.541 filas idénticas a alguna de train).
Se descarta `id` (es el número de fila de un archivo ordenado por clase).

## Condiciones de uso

Uso académico/investigación. Citar: Moustafa, N. & Slay, J. (2015). *UNSW-NB15: a comprehensive
data set for network intrusion detection systems (UNSW-NB15 network data set)*. Military
Communications and Information Systems Conference (MilCIS), IEEE.
