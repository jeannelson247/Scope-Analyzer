#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYI="${PYINSTALLER:-./venv/bin/pyinstaller}"
if [[ ! -x "$PYI" ]]; then
  PYI="$(command -v pyinstaller || true)"
fi
if [[ -z "$PYI" ]]; then
  echo "PyInstaller not found. Create/activate the project venv or run: ./venv/bin/python -m pip install pyinstaller" >&2
  exit 127
fi
"$PYI" -y "scope_analyzer/mac_mlx/ScopeAnalyzerMacMLX.spec"
