#!/usr/bin/env bash
set -euo pipefail
# Linux/WSL 验证入口：切换项目根目录，用源码包运行语法检查、测试及证据生成。

# Run from WSL/Linux with the project environment, never from the Windows
# global Python installation.  Plugin autoload is disabled because unrelated
# system pytest plugins can inject incompatible Hydra/ANTLR versions.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
# 未设置测试替身；依赖真实编码器的测试要求当前 Python 环境已准备好本地模型缓存。
python -m compileall -q src tests
python -m pytest -q
python scripts/verify_wsl.py
