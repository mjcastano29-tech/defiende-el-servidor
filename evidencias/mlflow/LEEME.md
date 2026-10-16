# Exportación de MLflow

- `runs.csv`: una fila por run (parámetros, métricas y tags). `conjunto=canonico` son los runs del commit `80cdc99` usados en la entrega; `reproduccion` es la re-ejecución del commit `450942b` con métricas idénticas.
- `runs/<run_id>.json`: registro completo de cada run (incluye datasets de entrada con su digest).
- `models/<experimento>__<run_id>/`: metadatos del modelo logueado y SHA-256 de `model.skops`.

Consulta sin servidor:

```python
import pandas as pd
runs = pd.read_csv('evidencias/mlflow/runs.csv')
runs[runs.conjunto == 'canonico'][['run_name', 'run_id', 'metric.val_recall', 'metric.val_fpr']]
```

La base completa (`mlflow/mlflow.db` + artefactos) se recupera con `dvc pull` y se abre con
`mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`.
