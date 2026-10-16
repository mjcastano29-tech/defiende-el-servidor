#!/usr/bin/env bash
# Alternativa a `dvc pull` para obtener los datos originales desde la fuente.
# Descarga las particiones publicadas de UNSW-NB15 en data/raw/ y verifica sus SHA-256
# (los mismos que publica el anexo del taller).
#
# Uso: scripts/download_data.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

# Copia fijada a un commit (mirror de terceros indicado por el docente).
# Verificado: idéntica byte a byte a la descarga oficial de UNSW (SharePoint, versión 2024-02-09).
GH="https://raw.githubusercontent.com/Nir-J/ML-Projects/44bbd02b9e70f74dd915218d5eb702f125341ffc/UNSW-Network_Packet_Classification"
for f in UNSW_NB15_training-set.csv UNSW_NB15_testing-set.csv; do
  [ -f "$RAW/$f" ] || curl -fL "$GH/$f" -o "$RAW/$f"
done

# Los archivos de documentación (features, list events, PDF) solo están en el SharePoint
# oficial de UNSW; se recuperan con `dvc pull`.
cd "$RAW"
sha256sum -c --ignore-missing "$ROOT/data/SHA256SUMS"
