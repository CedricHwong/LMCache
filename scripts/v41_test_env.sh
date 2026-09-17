#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
: <<'DOC'
Reproducible offline test entry for LMCache on a GPU-less host.

This is the phase-1 "env-harness" entry point: it prepares a throwaway
virtualenv and runs the *Python-side* LMCache logic tests either against the
pure-Python shim (``scripts/v41_stub_lmcache_native.py``) or against the real
C++ ``lmcache_native`` extension built locally with ``NO_GPU_EXT=1`` (no CUDA
on macOS/Apple Silicon, the non-GPU part compiles fine).

Usage
-----
  scripts/v41_test_env.sh [MODE] [pytest args...]

Modes
-----
  stub    (default) register the pure-Python shim as ``lmcache.lmcache_native``
          and run the Python-side logic tests offline (no build required).
  native  install torch, compile the real ``lmcache_native`` C++ extension
          (``NO_GPU_EXT=1 python setup.py build_ext --inplace``), then run the
          test suite against the compiled module.
  parity  cross-check the shim against a built native module
          (``v41_stub_lmcache_native.py --parity``); builds native first if
          it is missing.

With no pytest args the default suite runs: the v4.1-relevant detection, spec
and classification tests under ``tests/v1/gpu_connector``, the native
storage-op tests under ``tests/v1/lmcache_native``, and the hybrid fold/unfold
tests under ``tests/v1/distributed``.  Add ``-- --smoke`` to run only the
blocks-first detection test.

Env knobs
---------
  V41_PY           Python interpreter used for the venv (default: ``python3``).
  V41_REUSE_VENV=1 Keep the existing ``$REPO_ROOT/.testenv`` without
                   reinstalling dependencies (fast repeat runs).
  V41_NO_PIP=1     Skip dependency installation entirely (assume they exist).
  V41_TORCH=1      Also install torch in ``stub`` mode.  Required by the
                   v4.1 phase-1/2 tests: they import
                   ``lmcache.integration.vllm.utils`` and
                   ``lmcache.v1.gpu_connector.kv_format`` at module scope, and
                   that chain needs a real ``torch.nn`` (the in-test torch
                   stub only covers dtype constants).  A CPU wheel is enough;
                   no GPU is involved.
  V41_TORCH_INDEX_URL
                   pip ``--index-url`` used when installing torch (e.g.
                   ``https://download.pytorch.org/whl/cpu`` in CI).
  V41_CONFCUTDIR   Override the ``--confcutdir`` passed to pytest (default:
                   ``tests/v1`` when all test paths live under ``tests/v1``).
DOC

set -euo pipefail

_help() {
  sed -n '/^: <<'"'"'DOC'"'"'$/,/^DOC$/p' "$0" | sed '1d;$d'
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="${1:-stub}"
case "$MODE" in
  stub|native|parity) shift || true ;;
  -h|--help)
    _help
    exit 0
    ;;
  *)
    echo "unknown mode: $MODE (expected stub|native|parity)" >&2
    exit 2
    ;;
esac

# --help passthrough for bare pytest help.
if [ "${1:-}" = "--" ]; then
  shift
fi

# ---------------------------------------------------------------------------
# Figure out the default test set.
# ---------------------------------------------------------------------------
SMOKE=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--smoke" ]; then
    SMOKE=1
  else
    ARGS+=("$a")
  fi
done

if [ "${#ARGS[@]}" -eq 0 ]; then
  if [ "$SMOKE" -eq 1 ]; then
    ARGS=(tests/v1/gpu_connector/test_blocks_first_detection.py)
  else
    ARGS=(
      tests/v1/gpu_connector/test_blocks_first_detection.py
      tests/v1/gpu_connector/test_kv_format_classification.py
      tests/v1/gpu_connector/test_kv_format_specs.py
      tests/v1/gpu_connector/test_concrete_shape.py
      tests/v1/gpu_connector/test_utils_shape_desc.py
      tests/v1/gpu_connector/test_normalize_per_layer_formats.py
      tests/v1/gpu_connector/test_kv_format_detection.py
      tests/v1/gpu_connector/test_per_layer_kv_tuple_format.py
      tests/v1/lmcache_native
      tests/v1/distributed/test_bitmap_ops.py
    )
  fi
fi


# ---------------------------------------------------------------------------
# Locate/select Python and create the venv.
# ---------------------------------------------------------------------------
PY_BIN="${V41_PY:-python3}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "ERROR: python '$PY_BIN' not found on PATH" >&2
  exit 2
fi

VENV="$REPO_ROOT/.testenv"
VENV_PY="$VENV/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "[env-harness] creating venv at $VENV"
  "$PY_BIN" -m venv "$VENV"
fi

RUN_PY="$VENV_PY"

# ---------------------------------------------------------------------------
# torch install helper: CPU wheels are enough, no GPU is touched.
# ---------------------------------------------------------------------------
install_torch() {
  if [ -n "${V41_TORCH_INDEX_URL:-}" ]; then
    "$VENV_PY" -m pip install --index-url "$V41_TORCH_INDEX_URL" torch
  else
    "$VENV_PY" -m pip install torch
  fi
}

# ---------------------------------------------------------------------------
# Install dependencies (skippable for repeat runs).
# ---------------------------------------------------------------------------
if [ "${V41_REUSE_VENV:-0}" != "1" ] && [ "${V41_NO_PIP:-0}" != "1" ]; then
  echo "[env-harness] installing base deps into $VENV"
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel >/dev/null
  "$VENV_PY" -m pip install \
    pytest numpy numba msgspec cachetools pyyaml requests \
    prometheus_client sortedcontainers "py-cpuinfo" psutil aiohttp \
    "pyzmq>=25" "grpcio>=1.78" "protobuf>=6.31.1,<7"
  if [ "$MODE" = "native" ] || [ "$MODE" = "parity" ]; then
    echo "[env-harness] installing torch (needed to build lmcache_native)"
    install_torch
  elif [ "${V41_TORCH:-0}" = "1" ]; then
    echo "[env-harness] installing CPU-only torch (V41_TORCH=1)"
    install_torch
  fi
fi

# ---------------------------------------------------------------------------
# Build the real native extension (native / parity modes).
# ---------------------------------------------------------------------------
NATIVE_SO="$(ls "$REPO_ROOT"/lmcache/lmcache_native.*.so 2>/dev/null | head -1 || true)"
if { [ "$MODE" = "native" ] || [ "$MODE" = "parity" ]; } && [ -z "$NATIVE_SO" ]; then
  echo "[env-harness] building lmcache_native (NO_GPU_EXT=1, no CUDA)"
  (
    cd "$REPO_ROOT"
    NO_GPU_EXT=1 "$RUN_PY" setup.py build_ext --inplace
  )
  NATIVE_SO="$(ls "$REPO_ROOT"/lmcache/lmcache_native.*.so 2>/dev/null | head -1 || true)"
  if [ -z "$NATIVE_SO" ]; then
    echo "ERROR: lmcache_native build produced no .so" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# --confcutdir: suppress the heavy root tests/conftest.py for tests/v1 tests.
# ---------------------------------------------------------------------------
CONFCUTDIR="${V41_CONFCUTDIR:-}"
if [ -z "$CONFCUTDIR" ] && [ "${#ARGS[@]}" -gt 0 ]; then
  all_under_v1=1
  for a in "${ARGS[@]}"; do
    case "$a" in
      tests/v1/*) ;;
      *) all_under_v1=0 ;;
    esac
  done
  if [ "$all_under_v1" -eq 1 ]; then
    CONFCUTDIR="tests/v1"
  fi
fi

PYTEST_ARGS=()
if [ -n "$CONFCUTDIR" ]; then
  PYTEST_ARGS+=(--confcutdir "$CONFCUTDIR")
fi
PYTEST_ARGS+=("${ARGS[@]}")

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
STUB="$SCRIPT_DIR/v41_stub_lmcache_native.py"

run_stub() {
  "$RUN_PY" - "$REPO_ROOT" "$STUB" "${PYTEST_ARGS[@]}" <<'PY'
import importlib.util
import sys

repo_root, stub_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo_root)

# Register the pure-Python shim as ``lmcache.lmcache_native`` BEFORE any
# lmcache.* module imports it (the compiled extension may be absent).
spec = importlib.util.spec_from_file_location("lmcache.lmcache_native", stub_path)
module = importlib.util.module_from_spec(spec)
sys.modules["lmcache.lmcache_native"] = module
spec.loader.exec_module(module)

import pytest

raise SystemExit(pytest.main(sys.argv[3:]))
PY
}

case "$MODE" in
  stub)
    echo "[env-harness] MODE=stub  shim=$(basename "$STUB")"
    echo "[env-harness] pytest args: ${PYTEST_ARGS[*]}"
    run_stub
    ;;
  native)
    echo "[env-harness] MODE=native  .so=$NATIVE_SO"
    echo "[env-harness] pytest args: ${PYTEST_ARGS[*]}"
    "$RUN_PY" -m pytest "${PYTEST_ARGS[@]}"
    ;;
  parity)
    echo "[env-harness] MODE=parity  .so=$NATIVE_SO"
    "$RUN_PY" "$STUB" --parity --repo-root "$REPO_ROOT"
    ;;
esac
