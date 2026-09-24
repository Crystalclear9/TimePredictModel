#!/usr/bin/env bash
set -euo pipefail

# Run from WSL/Linux with the project environment, never from the Windows
# global Python installation.  Plugin autoload is disabled because unrelated
# system pytest plugins can inject incompatible Hydra/ANTLR versions.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m compileall -q src tests
python -m pytest -q
python scripts/verify_wsl.py
