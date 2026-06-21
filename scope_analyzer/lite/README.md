# Scope Analyzer Lite

Primary open-source/student build. Bundles the web UI, Python runtime,
deterministic analysis tools, presets, examples, NumPy/SciPy/Pandas, and
pywebview. It does **not** bundle model weights.

Developer build:

```bash
cd ~/Desktop/scope_studio03
source venv/bin/activate
pip install pywebview pyinstaller
./scope_analyzer/lite/build.sh
open dist/ScopeAnalyzerLite.app
```
