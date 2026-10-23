#!/usr/bin/env bash
# Prueba obligatoria de recuperación en una carpeta y un entorno NUEVOS, sin cachés previas.
#
#   scripts/recuperacion.sh [DESTINO] [REVISION_GIT]
#
# 1. Clona el repositorio en DESTINO (por defecto un directorio temporal) en REVISION_GIT (por defecto main)
# 2. Crea un entorno virtual nuevo e instala requirements.txt sin caché de pip
# 3. Recupera datos originales y preparados con DVC desde el remoto de Hugging Face (caché DVC nueva)
#    y verifica sus SHA-256
# 4. Descarga v1 y v2 de Hugging Face por SHA completo (caché HF nueva), predice sobre las 20 filas fijas
#    de validación y compara con las salidas guardadas
# El log completo queda en DESTINO/recuperacion.log y los resultados en DESTINO/evidencias/recuperacion/.
set -euo pipefail

REPO_URL="https://github.com/mjcastano29-tech/defiende-el-servidor.git"
DEST="${1:-$(mktemp -d)/defiende-el-servidor}"
REV="${2:-main}"
LOG="$(dirname "$DEST")/recuperacion.log"
mkdir -p "$(dirname "$DEST")"
exec > >(tee "$LOG") 2>&1

step() { echo; echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) · $* ==="; }

step "0. Entorno"
uname -srm; python3 --version; git --version
echo "Destino: $DEST · revisión: $REV"

step "1. Código: git clone + checkout $REV"
git clone --quiet "$REPO_URL" "$DEST"
cd "$DEST"
git checkout --quiet "$REV"
echo "commit: $(git rev-parse HEAD)"

step "2. Entorno virtual nuevo + dependencias fijadas (sin caché de pip)"
if python3 -m venv .venv 2>/dev/null; then :; else
  # Sistemas sin python3-venv: venv sin pip + get-pip.py
  rm -rf .venv && python3 -m venv --without-pip .venv
  curl -fsSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python - --quiet
fi
.venv/bin/pip install --quiet --no-cache-dir -r requirements.txt
.venv/bin/python --version; .venv/bin/pip --version
export HF_HOME="$DEST/.hf-home"          # caché de Hugging Face nueva y aislada
export MLFLOW_DISABLE_AGENT_HINT=1 HF_HUB_DISABLE_PROGRESS_BARS=1

step "3. Datos: dvc pull desde el remoto (caché DVC nueva en $DEST/.dvc/cache)"
.venv/bin/dvc remote list
.venv/bin/dvc pull data/raw.dvc data/processed.dvc
(cd data/raw && sha256sum -c ../SHA256SUMS)
.venv/bin/python scripts/compare_recovery.py data

step "4-5. Modelos: descarga de v1 y v2 por SHA, predicción y comparación"
.venv/bin/python scripts/compare_recovery.py models

step "Fin"
echo "Log: $LOG"
