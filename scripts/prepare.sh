#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
USER_ARGS=("$@")
SAM3D_ROOT_VALUE="${SAM3D_ROOT:-}"
SAM3D_CONFIG_VALUE="${SAM3D_PIPELINE_CONFIG:-}"
ROOT_IN_ARGS=0
CONFIG_IN_ARGS=0
SHOW_HELP=0

for ((index = 0; index < ${#USER_ARGS[@]}; index++)); do
    argument="${USER_ARGS[index]}"
    case "${argument}" in
        -h|--help)
            SHOW_HELP=1
            ;;
        --sam3d-root)
            ROOT_IN_ARGS=1
            if ((index + 1 < ${#USER_ARGS[@]})); then
                SAM3D_ROOT_VALUE="${USER_ARGS[index + 1]}"
            fi
            ;;
        --sam3d-root=*)
            ROOT_IN_ARGS=1
            SAM3D_ROOT_VALUE="${argument#*=}"
            ;;
        --sam3d-config)
            CONFIG_IN_ARGS=1
            if ((index + 1 < ${#USER_ARGS[@]})); then
                SAM3D_CONFIG_VALUE="${USER_ARGS[index + 1]}"
            fi
            ;;
        --sam3d-config=*)
            CONFIG_IN_ARGS=1
            SAM3D_CONFIG_VALUE="${argument#*=}"
            ;;
    esac
done

if [[ -z "${SAM3D_ROOT_VALUE}" ]]; then
    for candidate in \
        "${PROJECT_ROOT}/external/sam-3d-objects" \
        "${PROJECT_ROOT}/../sam-3d-objects"; do
        if [[ -d "${candidate}/sam3d_objects" ]]; then
            SAM3D_ROOT_VALUE="${candidate}"
            break
        fi
    done
fi

if [[ -z "${SAM3D_CONFIG_VALUE}" && -n "${SAM3D_ROOT_VALUE}" ]]; then
    for candidate in \
        "${SAM3D_ROOT_VALUE}/checkpoints/hf-download/checkpoints/pipeline.yaml" \
        "${SAM3D_ROOT_VALUE}/checkpoints/hf/checkpoints/pipeline.yaml" \
        "${SAM3D_ROOT_VALUE}/checkpoints/hf/pipeline.yaml"; do
        if [[ -f "${candidate}" ]]; then
            SAM3D_CONFIG_VALUE="${candidate}"
            break
        fi
    done
fi

SAM3D_PYTHON_BIN="${SAM3D_PYTHON:-}"
if [[ -z "${SAM3D_PYTHON_BIN}" && -n "${CONDA_PREFIX:-}" ]]; then
    SAM3D_PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${SAM3D_PYTHON_BIN}" && ${SHOW_HELP} -eq 1 ]]; then
    if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
        SAM3D_PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
    else
        SAM3D_PYTHON_BIN="python3"
    fi
fi

if [[ -z "${SAM3D_PYTHON_BIN}" || ( "${SAM3D_PYTHON_BIN}" != "python3" && ! -x "${SAM3D_PYTHON_BIN}" ) ]]; then
    echo "ERROR: set SAM3D_PYTHON to the official SAM3D environment interpreter." >&2
    echo "Example: export SAM3D_PYTHON=/path/to/conda/envs/sam3d-objects/bin/python" >&2
    exit 2
fi
if [[ ${SHOW_HELP} -eq 0 && ( -z "${SAM3D_ROOT_VALUE}" || ! -d "${SAM3D_ROOT_VALUE}/sam3d_objects" ) ]]; then
    echo "ERROR: SAM3D_ROOT does not contain sam3d_objects: ${SAM3D_ROOT_VALUE:-<unset>}" >&2
    exit 2
fi
if [[ ${SHOW_HELP} -eq 0 && ( -z "${SAM3D_CONFIG_VALUE}" || ! -f "${SAM3D_CONFIG_VALUE}" ) ]]; then
    echo "ERROR: SAM3D pipeline.yaml was not found: ${SAM3D_CONFIG_VALUE:-<unset>}" >&2
    echo "Set SAM3D_PIPELINE_CONFIG or pass --sam3d-config." >&2
    exit 2
fi

INJECTED_ARGS=()
if [[ ${ROOT_IN_ARGS} -eq 0 && -n "${SAM3D_ROOT_VALUE}" ]]; then
    INJECTED_ARGS+=(--sam3d-root "${SAM3D_ROOT_VALUE}")
fi
if [[ ${CONFIG_IN_ARGS} -eq 0 && -n "${SAM3D_CONFIG_VALUE}" ]]; then
    INJECTED_ARGS+=(--sam3d-config "${SAM3D_CONFIG_VALUE}")
fi

export LIDRA_SKIP_INIT=true
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"
exec "${SAM3D_PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prepare.py" \
    "${INJECTED_ARGS[@]}" "${USER_ARGS[@]}"
