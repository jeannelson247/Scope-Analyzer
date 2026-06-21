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
# Regenerate bundled toolbox examples so the packaged app ships them.
PYGEN="${PYTHON:-./venv/bin/python}"; [ -x "$PYGEN" ] || PYGEN="$(command -v python3)"
"$PYGEN" scripts/generate_lite_toolbox_examples.py

"$PYI" -y "scope_analyzer/lite/ScopeAnalyzerLite.spec"
