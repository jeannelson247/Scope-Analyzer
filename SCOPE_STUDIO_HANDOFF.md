# Scope Studio — handoff (resume in a separate project)

Purpose: pick up the Scope Studio / "scope analyzer" work later without re-deriving
context. Written 2026-06-20. Keep this file updated as the app evolves.

> Note on scope: the Orchestra AI-gym work is a DIFFERENT project, canonical at
> `~/Desktop/Orchestra`. (An origin copy of it also lives here under
> `orchestrator_lab/` — ignore it for Scope Studio work.) This doc is ONLY about
> Scope Studio so we can focus the other thread on Orchestra.

---

## What Scope Studio is
A macOS desktop app for scientists: load oscilloscope/instrument CSVs, view/transform
channels, do deterministic signal analysis, and export Nature-quality figures, with a
**local AI side chat** (MLX/Ollama/llama.cpp) that can call deterministic tools but
never computes the numbers itself.

- Stack: **PySide6 (Qt) + pyqtgraph**, NumPy/SciPy/pandas. Entry point `app.py`
  (~3,000 lines, class `MainWindow`). Theme in `style.qss`.
- Original CSV is never mutated; all transforms live in RAM/session state.

---

## Current state (2026-06-20)
- **Native app works.** `app.py` runs; 3-pane layout: controls (`controls_scroll`) |
  workspace tabs (`workspace_tabs`) | AI panel (`ai_panel`), in a `QSplitter`.
- **Theme refreshed.** `style.qss` was rewritten to the Claidev2 dark theme
  (Wong-blue `#2b8fd6` accent, card group-boxes, lifted/bright tabs, thin scrollbars).
  Original saved as `style.qss.bak.*`. **Needs a visual check** (launch + eyeball);
  revert by restoring the .bak if anything reads wrong.
- **A native collapsible-AI-rail experiment was REVERTED.** Hand-porting redesign
  structural features into Qt broke the layout twice (QSplitter min-width fights).
  Lesson recorded below — do not retry the native structural port.

---

## The GUI decision (important)
**Do NOT re-implement the redesign in PySide6.** Reason: the Claude-Design redesign
already exists and *works in a browser*; Qt is imperative and fights you (manual
splitter sizes vs `min-width`), and it can't be visually verified headless. Chosen
path: **make the HTML the front-end, wire it to the real Python backend.**

### Redesign assets (`Claidev2/`)
- `Scope Studio.dc.html` — editable Claude-Design source (real HTML/CSS, `<x-dc>` +
  `support.js` runtime). Open in a browser to view/edit the template + `class Component`.
- `Scope Studio (standalone).html` — self-unpacking offline build (a loader bundle,
  not readable source).
- `Scope Studio - Feature Log.html` — the redesign spec. v0.2 features:
  3-zone window; **density modes** (Beginner/Standard/Advanced); **channel cards**
  (replace the 8-col table; add/remove/select, color chip, column dropdown, L/R pill);
  **Tools & libraries** dropdown with per-tool deterministic panels; **collapsible AI
  rail**; dual-axis plot; publication export. Palette: bg `#1c1d21`/`#25262b`/`#34373e`,
  text `#e8e9ec`/`#9a9ca3`, accent `#2b8fd6`, Okabe-Ito series.
- `uploads/T0026.CSV` — the real example capture (125,000 rows × 5 cols:
  TIME, CH1, CH1 Peak Detect, CH2, CH2 Peak Detect).

### Bridge foundation already built (`scope_web/`)  ← start here
- `backend_api.py` — Python↔JS bridge (`Api` class). Reuses the REAL `csv_loader`
  (+ `minmax_decimate`); returns decimated JSON-able series. **Verified headless on
  T0026.CSV** (`python scope_web/backend_api.py` → PASS). Methods: `pick_csv`,
  `load_csv`, `column_stats`; stubs `list_models`, `chat`.
- `app_web.py` — pywebview desktop shell exposing `Api` as `window.pywebview.api`.
- `index.html` — minimal real UI (Open CSV → channel toggles → canvas plot + stats)
  in the Claidev2 palette. The seam onto which the full design folds.
- `README.md` — run steps + roadmap.

Run the web frontend:
```bash
pip install pywebview          # one-time; uses OS WebView, no Chromium
python scope_web/app_web.py
```

### Roadmap to finish the web frontend (in order)
1. Fold the `Claidev2/Scope Studio.dc.html` template + styles onto the `scope_web`
   bridge (density modes, channel cards, Tools dropdown, collapsible rail — already
   built in HTML), keeping every control wired to `window.pywebview.api`.
2. Plot upgrade: swap the canvas line plot for **dual-axis + zoom/pan** (uPlot,
   bundled locally) to match pyqtgraph interactions.
3. Wire deterministic tools into `Api`: `detect_anomalies`, `saturation_recovery`,
   `rlc_reconstruct`, column stats (all "no LLM").
4. Wire AI chat: `Api.chat()` → `ai_assistant.ask_model` (MLX backend), model list
   from `model_catalog`.
5. Package: PyInstaller (pywebview hooks already vendored in the venv).

The native `app.py` keeps working throughout; `scope_web/` grows beside it until ready.

---

## Backend module map (reuse these; don't reinvent)
- `csv_loader.py` — `load_csv(path) -> LoadedData(df, columns, units, meta)`;
  `minmax_decimate(x, y, target)` for plot downsampling.
- `signal_tools.py` — formula engine (`evaluate_formula`), `lowpass`, etc.
- `detect_anomalies.py` — anomaly scan.
- `saturation_recovery.py` — simple saturation/peak estimate (rise/droop projection).
- `rlc_reconstruct.py` — censored-ML RLC reconstruction (see below).
- `calibration.py` — `fit_forced_origin_gain` (V→A reference calibration).
- `ai_assistant.py` — `ask_model(prompt, model, backend, ...)`; MLX/Ollama/llama.cpp.
  NOTE its MLX resolver looks under `~/models/mlx` or `ScopeStudioModels/mlx`, NOT
  `/Volumes/JeanDrive1/Models/mlx`; point it via env or load by full path.
- `chat_actions.py` — action schema/routing for the side chat.
- `model_catalog.py` — model profiles/tiers.

---

## Domain knowledge worth keeping: the RLC reconstruction
The example shows a current pulse that the sensor **clips at ~6 kA** (2–5 ms). The
"Reconstruct RLC" tool recovers the true peak through the clipped region.

- Model (`rlc_reconstruct.py`): overdamped series-RLC discharge,
  `I(t) = A·(e^{-(t-t0)/τd} − e^{-(t-t0)/τr})`, 4 interpretable params.
- **"ML" = Maximum Likelihood, NOT machine learning.** It's a gray-box physics fit,
  deterministic NumPy/SciPy.
- **Censored-ML (Tobit-style) is the key idea:** clipped samples aren't real
  readings — they're lower bounds (true ≥ clip). Clean samples → least-squares
  residuals; censored samples → a one-sided hinge that fires only if the model dips
  BELOW the clip. So the plateau *constrains* the fit instead of biasing it.
  - Plain RLC fit (OLS, or dropping clipped samples) → underestimates / unconstrained
    peak. Censored-ML → unbiased true-peak recovery.
- Extras: log-parameterized (A>0, τd>τr by construction); optional fit window
  (free-discharge only); optional second sensor (Pearson) as extra clean data;
  residual-bootstrap → 95% band + peak CI. The overlay is the green dashed curve.

---

## How to run / verify
```bash
# native app
python app.py
# web frontend (after pip install pywebview)
python scope_web/app_web.py
# backend bridge headless check (no GUI)
python scope_web/backend_api.py
```

## Constraints / cautions
- Models live on the external SSD `/Volumes/JeanDrive1/Models/mlx`; MLX is Apple-only.
- Don't retry the native Qt structural port — go web-frontend.
- `style.qss` dark theme is unverified visually; `.bak` is the rollback.
- Other context files here: `README.md`, `CHANGELOG.md`,
  `DEVELOPMENT_LOG_SCOPE_STUDIO.txt`, `CLAUDE_GUI_CONTEXT_FULL.txt`.
