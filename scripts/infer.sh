#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MOD_PYTHON_BIN="${MOD_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

if [[ ! -x "${MOD_PYTHON_BIN}" ]]; then
    echo "ERROR: MOD Python is not executable: ${MOD_PYTHON_BIN}" >&2
    echo "Run 'bash scripts/install.sh' or set MOD_PYTHON=/path/to/python." >&2
    exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"
exec "${MOD_PYTHON_BIN}" "${PROJECT_ROOT}/scripts/infer.py" "$@"
