#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
INSTALL_EXTRAS="${MOD_INSTALL_EXTRAS:-train,iou}"
TORCH_INDEX_URL="${MOD_TORCH_INDEX_URL:-}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
if [[ -n "${TORCH_INDEX_URL}" ]]; then
    "${VENV_DIR}/bin/python" -m pip install \
        'torch>=2.5,<2.6' \
        --index-url "${TORCH_INDEX_URL}"
fi
"${VENV_DIR}/bin/python" -m pip install -e "${PROJECT_ROOT}[${INSTALL_EXTRAS}]"

echo "Environment installed at ${VENV_DIR}"
echo "Installed extras: ${INSTALL_EXTRAS}"
echo "Activate it with: . ${VENV_DIR}/bin/activate"
