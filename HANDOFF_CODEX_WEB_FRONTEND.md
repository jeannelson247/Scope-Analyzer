# Scope Studio — Codex handoff (web frontend + bridge)

Written 2026-06-21. Pick up the web-frontend work without re-deriving context.
Canonical project: `~/Desktop/scope_studio03`. Remote: `github.com/jeannelson247/Scope-Analyzer` (branch `main`).
Record every change in `DEVELOPMENT_LOG_SCOPE_STUDIO.txt` (CS-numbered) — do not skip.

## What this app is
macOS desktop app for scientists: load oscilloscope CSVs, transform channels, run
deterministic signal analysis, export figures, with a local-AI side chat that
**routes to deterministic tools but never computes numbers itself** (tested invariant —
`tests/test_llm_action_safety.py`). Stack: Python + NumPy/SciPy/pandas. Two front-ends:
native `app.py` (PySide6, ~3k LOC, works) and the new web frontend in `scope_web/`.
Strategy: native is maintenance-mode; **web is becoming primary**, then PyInstaller package.

## State as of this handoff (commit 8a3c8d3)
- `git status` clean except intentionally-untracked `CLAUDE_GUI_CONTEXT_FULL.txt`. All work pushed.
- Recent commits: CS70 RLC trusted-windows → de-risk/ignore → CS73 bridge contract test →
  CS74 web frontend → CS75 BBCM CH1 preset.
- Full test suite: **59 passing** (`python3 -m pytest -q`, run from repo root in `venv`).

## The bridge (start here) — `scope_web/`
- `backend_api.py` — `Api` class exposed to JS as `window.pywebview.api`. The browser is the
  view; Python does the work, reusing the real tested modules. **Importable headlessly**
  (`python3 scope_web/backend_api.py` → selftest PASS).
  - WIRED + contract-tested: `pick_csv()`, `load_csv(path)`, `column_stats(column, t_start?, t_end?)`.
  - Contract frozen by `tests/test_backend_api.py` (8 tests). **If you change a return shape,
    update that test in the same commit.**
  - STUBS (honest, return `{"ok":false,...}` or `note`): `chat(prompt,...)`, `list_models()`.
- `app_web.py` — pywebview shell; loads `index.html`, injects `Api`. Run:
  `pip install pywebview && python scope_web/app_web.py`.
- `index.html` — self-contained Claude-Design UI (**zero runtime CDN deps** — keep it that way
  for PyInstaller). 3-zone layout, density modes, channel cards, Tools dropdown, dual-axis
  canvas plot (saturation band + dashed RLC overlay), collapsible AI rail. Opening it in a
  plain browser auto-loads a synthetic T0026 demo (CH2 clips at 6 kA, RLC recovers ~8.4 kA).
  Live Statistics tool calls `column_stats` and prints a caption-ready string.

## Contract shapes (do not break silently)
- `load_csv` → `{ok, path, name, columns, x_col, y_cols, units, n_rows, series:{col:{x:[],y:[]}}}`
  (series are min-max **decimated** to ≤ `PLOT_POINTS+4`; peaks/troughs preserved).
- `column_stats` → `{ok, column, n, n_finite, min, max, mean, std, median, p5, p95, rms, window}`
  (`window` = `[t0,t1]` the stats covered, or null).

## NEXT TASK (highest leverage): wire channel-processing into the bridge
Goal: make the existing presets (incl. the new BBCM CH1 one) apply in the web UI, and feed
real data to the RLC overlay. Reuse native modules — invent nothing.
1. `Api.list_presets()` → return `presets.json`.
2. `Api.apply_channel(column, {gain, offset, formula})` → run `signal_tools.evaluate_formula`
   (sig: `evaluate_formula(formula, x, t_s, ...)`; helpers incl. `lowpass(y, t, cutoff_hz)`),
   apply gain/offset, return a decimated series in the same shape as `load_csv`'s `series`.
   Add this to `tests/test_backend_api.py` (freeze the shape; assert source df unchanged).
3. `index.html` channel cards: add a preset dropdown + (Advanced density) gain/offset/formula
   fields; on change call `apply_channel` and redraw. Currently cards have enable/colour/
   column/L-R/remove only.
4. Then wire RLC: `Api.reconstruct_rlc(...)` → `rlc_reconstruct.fit_rlc` (supports
   `trusted_windows`, censored sat_level, 95% CI). Render its curve as the overlay and surface
   `peak = X kA (95% CI a–b)` as a copyable caption near the saturation band.
5. Then `chat`→`ai_assistant.ask_model` (MLX; resolver looks under `~/models/mlx`, NOT the
   external SSD — pass full path/env), `list_models`→`model_catalog`.
6. Package: PyInstaller (pywebview hooks vendored in venv).

## BBCM CH1 recipe (already added — CS75, `presets.json`)
`"CH1 BBCM H-inverted + 10 kHz LP (gain 4)"`: gain 4.0, unit A,
formula `lowpass((2.5 - x) * 750, t, 1e4)`. = H-inverted busbar (center 2.5 V, 750 A/V) +
10 kHz LP + 4-module gain → ~6 kA full scale, comparable to CH2 Pearson. Verified through the
real engine. CH1 = busbar (BBCM) monitor; CH2 = Pearson; CH3 = Rogowski (raw V). User has
"no preference" on center 2.5-vs-2.52 / baseline — leave as is unless asked.

## UI/UX fixes the user will validate from a screenshot (do not pre-implement; await direction)
Axis titles + units and color-coded ticks per axis; cursor Y-readout (crosshair); label the
saturation band + RLC peak-CI caption; "pending" affordance on not-yet-wired tools; keep the
legend off the data. (The user is reviewing the rendered UI and will send a screenshot.)

## Guardrails
- Never mutate the source CSV. Keep `index.html` dependency-free. Keep the "LLM routes,
  deterministic tools compute" boundary. Run `python3 -m pytest -q` before each commit and log
  a CS entry. Git from a real terminal (the sandbox can leave a stale `.git/index.lock`).
- Key modules: `csv_loader.py`, `signal_tools.py`, `rlc_reconstruct.py`, `saturation_recovery.py`,
  `detect_anomalies.py`, `calibration.py`, `ai_assistant.py`, `chat_actions.py`, `model_catalog.py`.
- Deeper context: `SCOPE_STUDIO_HANDOFF.md`, `PROJECT_ANALYSIS_THREE_LENS.md`, `docs/METHODS.md`.
