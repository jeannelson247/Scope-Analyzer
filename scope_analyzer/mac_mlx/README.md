# Scope Analyzer Mac MLX

Apple Silicon build. Same web UI and deterministic tools as Lite, plus MLX
runtime libraries for optional local-assistant inference. Model weights are
still external/user-selected to keep releases small and license-clean.

Developer build:

```bash
cd ~/Desktop/scope_studio03
source venv/bin/activate
pip install pywebview pyinstaller -r requirements-mlx-mac.txt
./scope_analyzer/mac_mlx/build.sh
open dist/ScopeAnalyzerMacMLX.app
```
