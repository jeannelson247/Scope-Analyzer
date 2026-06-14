# Choose your version

Scope Studio ships as **one project, three ways to run it**. All three
share the same deterministic math (CSV loading, calibration formulas,
filters, anomaly detection, RLC reconstruction, 3D surfaces) - the AI
never computes numbers in any of them. Pick based on your machine, not
your needs: every mode (2D scope view, 3D sequential-pulse overlay, IV
surfaces, anomaly scan) is available in all three.

## 1. PulseLab (web, no install, no LLM)

**For:** anyone, on any OS, who just wants to plot a CSV right now.

- Open the hosted page (GitHub Pages) or `pulselab/index.html` directly
  in a browser - nothing to install.
- Drag-and-drop a scope CSV. Tabs/dropdown switch between 2D traces, 3D
  pulse overlays, and IV surfaces.
- Pure JavaScript (`pulselab_core.js`), runs identically in the browser
  and in Node (used for the backtest against real shot data).
- No AI chat panel - this is the "minimum requirement" variant.

## 2. Scope Studio Lite (desktop, small/no LLM)

**For:** students on a laptop (Windows, Mac, or Linux) who want the full
desktop app, including the AI side-chat, with a small footprint.

- Download the `ScopeStudio` build for your OS from the Releases page
  (built by `.github/workflows/build-desktop.yml`).
- Same PySide6 UI as Full, but the AI chat defaults to the lightweight
  model tier from `model_catalog.py` (<=2 GB router, e.g. Llama-3.2-1B/3B
  via Ollama). No MoEs, no MLX dependency.
- Install from source instead: `pip install -r requirements.txt && python app.py`.

## 3. Scope Studio Full (Mac, Apple Silicon, MLX)

**For:** the primary development machine - an Apple Silicon Mac with
room for larger local models.

- `ScopeStudioFull.app` bundles the MLX direct backend.
- Install from source: `pip install -r requirements.txt -r requirements-mlx-mac.txt && python app.py`.
- Full model lineup from `model_catalog.py` (Qwen3.5 4B/9B, DeepSeek-R1
  distill, etc.) - see `docs/MODEL_SELECTION_GUIDE.md`.

## What's the same everywhere

- **Modes, as tabs + a dropdown selector**: 2D oscilloscope plot, 3D
  shot-sequence / pulse overlay, IV (V-I) surface view, anomaly scan,
  RLC reconstruction.
- **Immutable raw data**: source CSVs are read-only; everything else is
  display state until you explicitly export.
- **Formula presets, calibration, filters**: identical formula language
  and MATLAB-derived presets (see `README.md`).

## What differs

| | PulseLab (web) | Lite (desktop) | Full (desktop, Mac) |
|---|---|---|---|
| Install | none | `pip install` or installer | `pip install` or installer |
| AI chat | none | small/local model | full model lineup, MLX |
| 3D / IV surfaces | yes (JS) | yes (matplotlib/OpenGL) | yes (matplotlib/OpenGL) |
| Best for | quick look, any OS | most students | Mac power users / dev |

If you're not sure: start with **PulseLab** to view your data today, and
install **Lite** when you want the AI assistant and full export options.
