#!/usr/bin/env bash
# Descarga UNSW-NB15 en data/raw/ y verifica checksums.
#
# Uso:
#   scripts/download_data.sh          # solo training/testing (fuente del profesor, GitHub)
#   scripts/download_data.sh --full   # además los 4 CSV completos y metadatos (SharePoint oficial UNSW)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

# 1) Training/testing: copia pública fijada a un commit (recomendada por el docente).
#    Verificado: idéntica byte a byte a la descarga oficial de UNSW (2024-02-09).
GH="https://raw.githubusercontent.com/Nir-J/ML-Projects/44bbd02b9e70f74dd915218d5eb702f125341ffc/UNSW-Network_Packet_Classification"
for f in UNSW_NB15_training-set.csv UNSW_NB15_testing-set.csv; do
  [ -f "$RAW/$f" ] || curl -fL "$GH/$f" -o "$RAW/$f"
done

# 2) Opcional: dataset completo desde el SharePoint oficial de UNSW (API REST, acceso anónimo).
if [ "${1:-}" = "--full" ]; then
  SHARE="https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS"
  BASE="https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au"
  DIR="/personal/z5025758_ad_unsw_edu_au/Documents/UNSW-NB15 dataset/CSV Files"
  JAR="$(mktemp)"; trap 'rm -f "$JAR"' EXIT
  curl -fsL -c "$JAR" -b "$JAR" -o /dev/null "$SHARE"   # obtiene la cookie de invitado
  for f in UNSW-NB15_1.csv UNSW-NB15_2.csv UNSW-NB15_3.csv UNSW-NB15_4.csv \
           NUSW-NB15_GT.csv NUSW-NB15_features.csv UNSW-NB15_LIST_EVENTS.csv \
           "The UNSW-NB15 description.pdf"; do
    [ -f "$RAW/$f" ] && continue
    path="$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$DIR/$f")"
    curl -fL -c "$JAR" -b "$JAR" -o "$RAW/$f" "$BASE/_api/web/GetFileByServerRelativeUrl('$path')/\$value"
  done
fi

# 3) Verificación de integridad (solo de los archivos presentes).
cd "$RAW"
sha256sum -c --ignore-missing "$ROOT/data/SHA256SUMS"
