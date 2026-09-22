#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_PYTHON="${MOD_PYTHON:-${SAM3D_PYTHON:-}}"
if [[ -z "${DEMO_PYTHON}" && -n "${CONDA_PREFIX:-}" ]]; then
    DEMO_PYTHON="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${DEMO_PYTHON}" ]]; then
    DEMO_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi
for argument in "$@"; do
    if [[ "${argument}" == "--help" || "${argument}" == "-h" ]]; then
        exec python3 "${PROJECT_ROOT}/scripts/demo.py" "$@"
    fi
done
if [[ ! -x "${DEMO_PYTHON}" ]]; then
    echo "ERROR: activate the SAM3D environment or set MOD_PYTHON to its Python." >&2
    echo "For the offline --cpu-smoke test, run bash scripts/install.sh first." >&2
    exit 2
fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${DEMO_PYTHON}" "${PROJECT_ROOT}/scripts/demo.py" "$@"
