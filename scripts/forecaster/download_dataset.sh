#!/usr/bin/env bash
# Download the csml_v2 paper corpus from Google Drive and unzip into
# data/csml_v2/raw_markdown/ — the layout that `load_papers_from_markdown`
# expects (<paper_id>/auto/<paper_id>.md).
#
# Idempotent: if the target directory already has the expected file count
# (~108k) we skip both the download and the unzip.

set -euo pipefail
cd "$(dirname "$0")/../.."

DRIVE_ID="${DRIVE_ID:-1Ztkvua-CvZKrgbHjb1dxwlXCi53NuGBv}"
DEST_DIR="${DEST_DIR:-data/csml_v2/raw_markdown}"
ZIP_PATH="${ZIP_PATH:-data/csml_v2.zip}"
MIN_FILES="${MIN_FILES:-100000}"

mkdir -p "$(dirname "${ZIP_PATH}")"
mkdir -p "${DEST_DIR}"

existing=$(find "${DEST_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l || echo 0)
if [[ "${existing}" -ge "${MIN_FILES}" ]]; then
  echo "[download_dataset] ${DEST_DIR} already has ${existing} paper directories — skipping."
  exit 0
fi

if [[ ! -s "${ZIP_PATH}" ]]; then
  echo "[download_dataset] gdown ${DRIVE_ID} -> ${ZIP_PATH}"
  python -m gdown "${DRIVE_ID}" -O "${ZIP_PATH}"
fi

echo "[download_dataset] Unzipping ${ZIP_PATH} -> ${DEST_DIR}"
(cd "${DEST_DIR}" && unzip -q -n "../../$(basename "${ZIP_PATH}")")

final=$(find "${DEST_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l || echo 0)
echo "[download_dataset] Done. ${final} paper directories in ${DEST_DIR}."
