# Integridad del test: congelación anterior a la evaluación

El test se evaluó **una sola vez**, el 2026-10-14, cuando regía la primera guía del taller (que todavía no
pedía v1/v2 ni Hugging Face). La guía actualizada llegó el 2026-10-16. Esta carpeta reúne los registros
fechados que muestran que el modelo y el umbral se eligieron **solo con validación** y quedaron congelados
**antes** de mirar el test, y que la guía nueva no cambió esa decisión: v2 es exactamente la elección congelada
(E6, run `b1caa19139d54c2ea08d8984bcbb0399`, umbral 0.6673794060957008).

## Línea de tiempo (UTC, 2026-10-14)

| Hora | Fuente | Evento |
|---|---|---|
| 22:52:53 | Git `86d0212` | Criterio de selección (máx. recall con FPR ≤ 5 % en validación) e hipótesis en `params.yaml`, antes de entrenar |
| 22:55:43 | Git `80cdc99` | Código con el que se ejecutó el pipeline |
| 22:56:00.520 | Prefect | Inicio del flujo `tangible-tody` (entrenamiento de E0–E6) |
| 22:58:26.324 | MLflow | Última métrica de validación del run elegido |
| 22:58:31.803 | Prefect | Fin de `comparar-y-congelar` (elección escrita en `seleccion.json`) |
| 22:58:31.808 | Prefect | Inicio de `evaluar-test` |
| 22:58:34.773 | MLflow | Primera métrica de test del run elegido |
| 23:01:45 | Git `548227a` | Commit de resultados (selección + test) |

## Archivos

| Archivo | Contenido | Cómo verificarlo |
|---|---|---|
| `linea_de_tiempo.json` | Eventos anteriores y tres comprobaciones automáticas (todas `true`) | `python scripts/export_freeze_evidence.py` |
| `mlflow_metricas_run_v2.csv` | Historial de cada métrica del run de v2 con su timestamp | `dvc pull mlflow.dvc` y `MlflowClient().get_metric_history("b1caa19139d54c2ea08d8984bcbb0399", "test_recall")` sobre `sqlite:///mlflow/mlflow.db` |
| `prefect_tareas.csv` | Estado, inicio y fin de las 12 tareas del flujo `tangible-tody` | Exportado de la base local de Prefect del equipo donde se ejecutó (no se versiona) |
| Commits Git | Fecha de cada commit | `git show -s --format='%H %cI' 86d0212 80cdc99 548227a` |

## Qué pasó después con el test

- 2026-10-16: una re-ejecución completa del pipeline (commit `450942b`) volvió a evaluar el **mismo** modelo
  elegido por el **mismo** criterio, con métricas idénticas (prueba de reproducibilidad, no una nueva elección).
- `reports/analisis_post_hoc.csv`: diagnóstico posterior de E1/E3/E4 en test para entender la subida de la FPR.
  Está declarado como posterior y **no** modificó la selección.
- `evidencias/metricas-test.json` y `predicciones-test.csv` se recalcularon con el modelo congelado de v2 y
  coinciden exactamente con la evaluación original.

No existe un commit de `seleccion.json` anterior al test: la elección y la evaluación ocurrieron en la misma
ejecución del pipeline, separadas por los registros de arriba. No se reconstruyó ni se fechó a posteriori
ningún commit.
