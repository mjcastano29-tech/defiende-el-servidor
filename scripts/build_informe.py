"""Genera informe.pdf (máx. 5 páginas) leyendo las cifras de evidencias/ y models/.

Requiere reportlab (solo para el informe, no para el modelo): pip install reportlab
Uso: python scripts/build_informe.py [salida.pdf]
"""
import csv
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidencias"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "informe.pdf"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("DV", FONT_DIR / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", FONT_DIR / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVM", FONT_DIR / "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")

GREEN, INK, MUTED, RULE = colors.HexColor("#1d4d35"), colors.HexColor("#1a1a1a"), colors.HexColor("#555555"), \
    colors.HexColor("#d9d9d4")
ss = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=ss["Normal"], fontName="DV", fontSize=8.6, leading=11.4, textColor=INK,
                      alignment=TA_JUSTIFY, spaceAfter=4)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=7.4, leading=9.4, alignment=0, spaceAfter=2)
CELL = ParagraphStyle("c", parent=SMALL, fontSize=7, leading=8.6, spaceAfter=0)
H1 = ParagraphStyle("h1", parent=BODY, fontName="DVB", fontSize=15, leading=19, textColor=GREEN, spaceAfter=2)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DVB", fontSize=10.5, leading=13, textColor=GREEN,
                    spaceBefore=7, spaceAfter=3)
SUB = ParagraphStyle("sub", parent=SMALL, textColor=MUTED, fontSize=8, leading=10.5)


def load(name):
    return json.loads((EV / name).read_text(encoding="utf-8"))


def pct(x, d=1):
    return f"{x * 100:.{d}f} %".replace(".", ",")


def num(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def thousands(n):
    return f"{int(n):,}".replace(",", ".")


def table(rows, widths, header=True, align_right_from=None):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3ef"))]
        for c in data[0]:
            c.style = ParagraphStyle("hc", parent=CELL, fontName="DVB")
    t.setStyle(TableStyle(style))
    return t


def main():
    audit, sel, test = load("auditoria.json"), load("seleccion.json"), load("metricas-test.json")
    rec = load("recuperacion/comparacion.json")
    exps = list(csv.DictReader((EV / "experimentos.csv").open(encoding="utf-8")))
    hf = json.loads((ROOT / "configs/hf_revisions.json").read_text(encoding="utf-8"))
    v = {k: json.loads((ROOT / f"models/{k}/config.json").read_text(encoding="utf-8")) for k in ("v1", "v2")}
    fam = json.loads((ROOT / "reports/test_metrics.json").read_text(encoding="utf-8"))["por_familia"]
    posthoc = list(csv.DictReader((ROOT / "reports/analisis_post_hoc.csv").open(encoding="utf-8")))
    tr, te = audit["archivos"]
    dtr, dte = audit["duplicados"]["train"], audit["duplicados"]["test"]
    part, cross = audit["particiones"], audit["cruces_entre_particiones"]
    m, cm_ = test["metricas"], test["matriz_confusion"]["valores"]
    sub = test["subconjunto_sin_entradas_vistas_en_train"]
    s = []
    P = lambda t, st=BODY: s.append(Paragraph(t, st))  # noqa: E731

    # ---------- Encabezado
    P("Defiende el servidor · Informe", H1)
    P("MLOps · Visualización 2 · 2026-II · <b>Grupo A</b> · Jorge Horacio Rojas Criollo y Maria Jimena Castaño "
      "Albarracin · 30 de octubre de 2026", SUB)
    P("Código: github.com/mjcastano29-tech/defiende-el-servidor (tags v1, v2) · Modelos: "
      "huggingface.co/Marimar2006/defiende-el-servidor · DVC: huggingface.co/datasets/Marimar2006/"
      "defiende-el-servidor-dvc · Índice de evidencias: entrega.md", SUB)
    P("Trabajo conjunto, desarrollado en clase desde el equipo de Maria Jimena: por eso los commits y los "
      "repositorios de Hugging Face aparecen en sus cuentas (mjcastano29-tech, Marimar2006).", SUB)
    s.append(Spacer(1, 4))
    P("<b>Objetivo.</b> Clasificar conexiones de UNSW-NB15 como normal (0) o ataque (1, clase positiva) y, sobre "
      "todo, versionar código, datos, experimentos y modelos de forma que otra persona pueda recuperar una "
      "versión concreta, saber de qué datos y experimento viene y usarla para predecir. Cadena de trazabilidad: "
      "<b>commit Git → referencia DVC → run de MLflow → revisión de Hugging Face</b>, enlazada por hashes.")

    # ---------- 1. Datos
    P("1. Datos, auditoría y limpieza", H2)
    P(f"Se usaron las particiones publicadas, descargadas el {audit['fuentes']['fecha_descarga']} del SharePoint "
      "oficial de UNSW y del mirror indicado por el docente; ambas copias son idénticas byte a byte y su SHA-256 "
      "coincide con el anexo del taller. Los CSV traen BOM (se leen con utf-8-sig).")
    s.append(table([
        ["Archivo", "Filas × col.", "Ataque", "Duplicados por entradas", "Grupos contradictorios", "NaN / '-' en service",
         "SHA-256 (inicio)"],
        [tr["archivo"], f"{thousands(tr['filas'])}×{tr['columnas']}", pct(tr["tasa_ataque"]),
         f"{thousands(dtr['filas_duplicadas'])} ({str(dtr['pct_duplicadas']).replace('.', ',')} %)",
         f"{dtr['grupos_con_etiqueta_contradictoria']} ({dtr['filas_en_grupos_contradictorios']} filas)",
         f"0 / {thousands(tr['faltantes_codificados'][chr(115) + 'ervice == ' + chr(39) + '-' + chr(39)])}",
         tr["sha256"][:16] + "…"],
        [te["archivo"], f"{thousands(te['filas'])}×{te['columnas']}", pct(te["tasa_ataque"]),
         f"{thousands(dte['filas_duplicadas'])} ({str(dte['pct_duplicadas']).replace('.', ',')} %)",
         f"{dte['grupos_con_etiqueta_contradictoria']} ({dte['filas_en_grupos_contradictorios']} filas)",
         f"0 / {thousands(te['faltantes_codificados'][chr(115) + 'ervice == ' + chr(39) + '-' + chr(39)])}",
         te["sha256"][:16] + "…"],
    ], [3.9, 1.8, 1.2, 2.6, 2.4, 2.0, 2.9]))
    s.append(Spacer(1, 4))
    P("<b>Variables.</b> 42 predictoras (39 numéricas y <i>proto</i>, <i>service</i>, <i>state</i>). Se excluyen "
      "<i>label</i> (respuesta), <i>attack_cat</i> (Normal ⇔ label 0: revela la respuesta) e <i>id</i>: es el "
      "número de fila 1…n de un archivo ordenado por clase, sin significado para el problema; solo se conserva "
      "para auditoría y para el <i>row_id</i> de las predicciones.")
    P("<b>Política de limpieza.</b> No se elimina ninguna fila, para no cambiar la población estudiada. Los "
      "duplicados (42 % del entrenamiento, 98 % de la familia Generic) y los grupos de entradas idénticas con "
      "etiqueta contradictoria se conservan; su riesgo es la fuga entre particiones, que se impide agrupando. "
      "<i>service='-'</i> (servicio no identificado) es una categoría propia. Categorías raras (&lt;0,1 %) o no "
      "vistas (<i>state</i> ACC/CLO solo aparecen en test) van a una columna «infrecuente». El efecto de "
      "deduplicar se midió como experimento (E5).")
    P(f"<b>Particiones.</b> Entrenamiento y validación salen solo del archivo de entrenamiento con "
      f"StratifiedGroupKFold (5 pliegues, semilla {part['semilla']}, primer pliegue = validación), usando como "
      f"grupo el hash de las 42 variables: {thousands(part['tamanos']['train'])} filas de entrenamiento y "
      f"{thousands(part['tamanos']['val'])} de validación, con {cross['grupos_compartidos_train_val']} grupos "
      f"compartidos. El test ({thousands(part['tamanos']['test'])} filas) no se toca: "
      f"{str(cross['pct_test_solapado']).replace('.', ',')} % de sus filas tiene entradas idénticas a filas de "
      "entrenamiento/validación; se reporta y se mide su efecto. Codificadores y escalado se ajustan solo con "
      "entrenamiento, dentro de un Pipeline de scikit-learn (log1p + StandardScaler en numéricas, OneHotEncoder "
      "en categóricas) que se guarda junto al estimador. DVC conserva dos versiones de datos: originales "
      f"(<i>data/raw</i>, md5 {part['dvc']['data/raw.dvc'][:8]}…) y preparados con sus particiones "
      f"(<i>data/processed</i>, md5 {part['dvc']['data/processed.dvc'][:8]}…); la asignación por fila está en "
      "asignacion_splits.csv.")

    # ---------- 2. Experimentos
    P("2. Experimentos (validación)", H2)
    P("Siete configuraciones de dos familias (lineal y árboles) más una referencia, con las mismas particiones "
      "y cambiando una decisión a la vez. Las hipótesis se escribieron en params.yaml antes de entrenar (commit "
      "86d0212). Cada run registra en MLflow hipótesis, parámetros, semilla, commit, datasets de entrada, "
      "métricas y el modelo completo. Política común de umbral: <b>máximo recall con FPR ≤ 5 % en "
      "validación</b>. Métricas con la clase ataque como positiva; si un denominador es 0, la métrica se reporta "
      "como 0.")
    rows = [["ID", "Familia · cambio", "Hipótesis (resumen)", "Umbral", "Recall", "Precisión", "FPR", "Macro-F1",
             "Run ID"]]
    hyp = {"E0": "Piso: no distingue clases", "E1": "Lineal basta parcialmente (AUC>0,95)",
           "E2": "Balancear solo mueve el umbral", "E3": "Árboles mejoran recall",
           "E4": "Sin TTL baja poco", "E5": "Duplicados aportan poco", "E6": "Más capacidad: mejora <1 punto"}
    chg = {"E0": "Dummy (prior)", "E1": "Regresión logística", "E2": "E1 + class_weight balanced",
           "E3": "HistGradientBoosting", "E4": "E3 sin sttl/dttl/ct_state_ttl", "E5": "E3 sin filas repetidas",
           "E6": "E3 con 500 it., 63 hojas, lr 0,05"}
    for e in exps:
        rows.append([f"<b>{e['id']}</b>" + (f" ({e['version_entregada']})" if e["version_entregada"] else ""),
                     chg[e["id"]], hyp[e["id"]], num(float(e["umbral"]), 3), num(float(e["recall"])),
                     num(float(e["precision"])), num(float(e["fpr"])), num(float(e["macro_f1"])),
                     e["run_id"][:8] + "…"])
    s.append(table(rows, [1.4, 3.4, 3.4, 1.3, 1.3, 1.6, 1.2, 1.6, 2.0]))
    s.append(Spacer(1, 4))
    P("<b>Lectura.</b> Todas las hipótesis se confirmaron. E2 cambió nuestra opinión: balancear clases parecía "
      "necesario por la prevalencia invertida, pero con el umbral elegido explícitamente no aporta (con umbral "
      "0,5 sí baja la FPR de 13,4 % a 8,1 %, solo porque desplaza el corte). El cambio de familia (E3) es lo que "
      "más aporta: +4,2 puntos de recall con la misma FPR y menor tiempo de entrenamiento. Quitar el TTL (E4), un "
      "artefacto del laboratorio, cuesta solo 0,4 puntos, y deduplicar (E5) cuesta 0,4 con 42 % menos filas. "
      "La comparación es justa porque comparte particiones, preprocesamiento, semilla y regla de umbral, y "
      "ningún grupo de entradas idénticas cruza entrenamiento y validación.")

    # ---------- 3. v1 / v2
    P("3. Versiones v1 y v2", H2)
    vv = [["", "v1", "v2"],
          ["Modelo", "E1 · regresión logística", "E6 · HistGradientBoosting"],
          ["Umbral", num(v["v1"]["threshold"], 6), num(v["v2"]["threshold"], 6)]]
    for k, lab in [("recall", "Recall val."), ("precision", "Precisión val."), ("fpr", "FPR val."),
                   ("macro_f1", "Macro-F1 val.")]:
        vv.append([lab, num(v["v1"]["validation_metrics"][k]), num(v["v2"]["validation_metrics"][k])])
    vv += [["Ataques no detectados / falsas alarmas (val.)",
            f"{thousands(v['v1']['validation_metrics']['fn'])} / {thousands(v['v1']['validation_metrics']['fp'])}",
            f"{thousands(v['v2']['validation_metrics']['fn'])} / {thousands(v['v2']['validation_metrics']['fp'])}"],
           ["Git (tag → commit)", "v1 → 57c2c870…", "v2 → ea3a50fd…"],
           ["MLflow run", v["v1"]["lineage"]["mlflow_run_id"], v["v2"]["lineage"]["mlflow_run_id"]],
           ["Hugging Face (SHA)", hf["v1"]["sha"], hf["v2"]["sha"]],
           ["SHA-256 model.skops", v["v1"]["model_sha256"], v["v2"]["model_sha256"]]]
    s.append(table(vv, [4.2, 6.6, 6.6]))
    s.append(Spacer(1, 4))
    P("<b>v1</b> es la primera solución funcional: preprocesamiento, regresión logística y umbral elegidos e "
      "identificados. <b>v2</b> es la revisión basada en los experimentos: cambia la familia de modelo y sus "
      "hiperparámetros, con los mismos datos, particiones y política de umbral. <b>Mejora</b> el recall en 4,7 "
      f"puntos (de {thousands(v['v1']['validation_metrics']['fn'])} a {thousands(v['v2']['validation_metrics']['fn'])} "
      "ataques no detectados en validación) y el macro-F1 de 0,911 a 0,945. "
      "<b>Empeora</b> la interpretabilidad (sin coeficientes directos) y el tamaño del artefacto (0,3 MB → 5,8 MB); "
      "la FPR no cambia porque ambos umbrales se fijan con la misma regla. <b>Errores aceptados</b>: ~5 % de "
      "conexiones normales alarmadas y ~4,8 % de ataques no detectados en validación. Cada versión se publicó "
      "en Hugging Face con el artefacto exacto del run de MLflow (mismo SHA-256), su config.json (umbral, regla, "
      "variables, métricas y linaje), dependencias y model card.")

    # ---------- 4. Umbral
    P("4. Elección del umbral", H2)
    P("Un ataque ignorado cuesta más que una falsa alarma, pero un equipo de seguridad no puede revisar una "
      "avalancha de alertas falsas. Por eso fijamos un presupuesto de falsas alarmas (5 % del tráfico normal) "
      "y, dentro de él, maximizamos los ataques detectados; el desempate es macro-F1. El umbral se calcula sobre "
      "la curva ROC de validación (mayor TPR con FPR ≤ 0,05). Con 0,5 la v2 detectaría 97,6 % de los ataques "
      "pero alarmaría 8,3 % del tráfico normal. La selección se congeló antes del test: en el flujo de Prefect, "
      f"la tarea comparar-y-congelar terminó a las "
      f"{sel['congelacion_previa_al_test']['prefect']['tareas']['comparar-y-congelar']['fin_utc'][11:23]} UTC y "
      f"evaluar-test empezó a las "
      f"{sel['congelacion_previa_al_test']['prefect']['tareas']['evaluar-test']['inicio_utc'][11:23]} UTC "
      "(2026-10-14), y el criterio estaba en Git desde el commit 86d0212.")

    # ---------- 5. Test
    P("5. Evaluación final en test (solo v2, una vez)", H2)
    s.append(table([
        ["", "Recall", "Precisión", "FPR", "Macro-F1", "ROC-AUC", "Matriz [[TN, FP], [FN, TP]]"],
        ["Validación (v2)", num(v["v2"]["validation_metrics"]["recall"]),
         num(v["v2"]["validation_metrics"]["precision"]), num(v["v2"]["validation_metrics"]["fpr"]),
         num(v["v2"]["validation_metrics"]["macro_f1"]), num(v["v2"]["validation_metrics"]["roc_auc"]),
         f"[[{thousands(v['v2']['validation_metrics']['tn'])}, {thousands(v['v2']['validation_metrics']['fp'])}], "
         f"[{thousands(v['v2']['validation_metrics']['fn'])}, {thousands(v['v2']['validation_metrics']['tp'])}]]"],
        [f"Test completo ({thousands(test['filas_evaluadas'])} filas)", num(m["recall"]), num(m["precision"]),
         num(m["fpr"]), num(m["macro_f1"]), num(m["roc_auc"]),
         f"[[{thousands(cm_[0][0])}, {thousands(cm_[0][1])}], [{thousands(cm_[1][0])}, {thousands(cm_[1][1])}]]"],
        [f"Test sin entradas vistas ({thousands(sub['filas'])})", num(sub["recall"]), num(sub["precision"]),
         num(sub["fpr"]), num(sub["macro_f1"]), "—", "—"],
    ], [4.0, 1.4, 1.5, 1.4, 1.5, 1.5, 6.1]))
    s.append(Spacer(1, 4))
    worst = sorted(fam, key=lambda f: f["marcadas_como_ataque"])
    fuzz = next(f for f in fam if f["attack_cat"] == "Fuzzers")
    P(f"El resultado <b>empeoró y se conserva</b>: el recall se mantiene (96,9 %), pero la tasa de falsas alarmas "
      f"sube de 5 % a {pct(m['fpr'])}. El solapamiento con entrenamiento no lo explica (sin esas filas, FPR "
      f"{pct(sub['fpr'])}). La causa es un cambio en el tráfico normal entre archivos: en test, 41 % de las "
      "conexiones normales tiene sttl=254 (20 % en entrenamiento) y son las que el modelo confunde. Reponderar "
      "la validación con esa mezcla ya duplica la FPR esperada (≈10,6 %). Un diagnóstico posterior, declarado y "
      "sin efecto en la selección, muestra que ningún modelo se habría salvado (FPR en test entre "
      f"{pct(min(float(r['posthoc_test_fpr']) for r in posthoc))} y "
      f"{pct(max(float(r['posthoc_test_fpr']) for r in posthoc))}, también sin TTL). La capacidad de ordenar "
      f"se conserva (ROC-AUC {num(m['roc_auc'])}); lo que no se traslada es el umbral. Los ataques no "
      f"detectados se concentran en Fuzzers ({pct(1 - fuzz['marcadas_como_ataque'], 0)} sin alerta; 1.257 de "
      "1.415). Las predicciones por fila (row_id, y_true, score_attack, y_pred) y las métricas se recalculan "
      "desde predicciones-test.csv y coinciden con la evaluación original.")
    s.append(KeepTogether([Image(str(ROOT / "imagenes_resultados/02_test.png"), width=15.5 * cm, height=5.7 * cm)]))

    # ---------- 6. Recuperación
    tl = json.loads((EV / "congelacion" / "linea_de_tiempo.json").read_text(encoding="utf-8"))
    P("Integridad del test", H2)
    P("El test se evaluó una sola vez, el 14 de octubre, cuando regía la primera guía; la guía actualizada "
      "(16 de octubre) no cambió la decisión: v2 es exactamente la elección congelada entonces. La secuencia "
      "está respaldada por registros fechados exportados en evidencias/congelacion/ y verificables en Git y en "
      "la base de MLflow recuperable con DVC:")
    s.append(table([["Hora UTC (2026-10-14)", "Fuente", "Evento"]] + [
        [e["hora_utc"][11:23], e["fuente"], e["evento"]] for e in tl["eventos"]], [3.0, 1.8, 12.6]))
    s.append(Spacer(1, 4))
    P("No existe un commit de seleccion.json anterior al test, porque elección y evaluación ocurrieron en la "
      "misma ejecución del pipeline; no se reconstruyó ni se fechó a posteriori ningún commit. La re-ejecución "
      "del 16 de octubre y el recálculo de las evidencias usan el mismo modelo congelado y dan cifras idénticas; "
      "el diagnóstico de otros modelos en test es posterior, está declarado y no modificó la selección.")

    P("6. Recuperación de versiones", H2)
    rv1, rv2 = rec["versiones"]["v1"], rec["versiones"]["v2"]
    P("En una carpeta y un entorno virtual nuevos, sin cachés de pip, DVC ni Hugging Face (scripts/"
      "recuperacion.sh): se clonó el código, se instalaron las dependencias fijadas, se recuperaron con "
      "<i>dvc pull</i> los datos originales y preparados desde el remoto público (huellas verificadas), se "
      "descargaron v1 y v2 por su SHA completo, se cargaron preprocesamiento, modelo y umbral, y se predijo "
      f"sobre {rec['entrada']['filas']} filas fijas de validación (10 normales y 10 ataques). Resultado: "
      f"<b>{rec['resultado']}</b>. v1: predicciones idénticas, máx. |Δscore| = {rv1['max_abs_diff_score']:.1g}; "
      f"v2: idénticas, máx. |Δscore| = {rv2['max_abs_diff_score']:.1g} (tolerancia {rec['tolerancia_abs_score']}). "
      "El SHA-256 del modelo descargado coincide con el registrado en MLflow, DVC y la model card.")

    # ---------- 7. Limitaciones
    P("7. Limitaciones", H2)
    P("UNSW-NB15 es tráfico sintético de laboratorio (2015): no demuestra eficacia en una red actual. La "
      "prevalencia de ataque (55–68 %) es irreal; con ataques raros la precisión caería mucho para la misma FPR. "
      "El TTL y las filas repetidas son huellas del laboratorio. El umbral depende del tráfico normal con el que "
      "se calibra y debe recalibrarse en la red de destino; el test muestra ese riesgo. El test se evaluó el "
      "14 de octubre con la primera guía; la selección se hizo solo con validación y no se cambió con la guía "
      "actualizada. Las variables ct_* requieren estado de las 100 conexiones previas para usarse en línea.")
    P("Referencias", H2)
    for ref in [
        "Moustafa, N. & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection "
        "systems. MilCIS, IEEE. https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825–2830.",
        "Documentación de DVC (dvc.org/doc), MLflow Tracking (mlflow.org/docs), Hugging Face Hub "
        "(huggingface.co/docs/hub: descarga por revisión y model cards).",
        "Uso de IA: Claude Code (Anthropic) como asistente de programación y redacción; ver entrega.md.",
    ]:
        P(ref, SMALL)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DV", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(1.6 * cm, 1.0 * cm, "MLOps 2026-II · Defiende el servidor · Grupo A")
        canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"{doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.4 * cm,
                            bottomMargin=1.5 * cm, title="Defiende el servidor · Informe",
                            author="Jorge Horacio Rojas Criollo; Maria Jimena Castaño Albarracin")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"{OUT} · {doc.page} páginas")


if __name__ == "__main__":
    main()
