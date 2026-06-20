# Scope Studio — web frontend (bridge foundation)

The intended redesign (`Claidev2/`) is a browser UI. Rather than re-implement it
in PySide6, we make the **HTML the front-end** and wire it to the **real Python
backend**. This folder is the proven foundation for that.

## What's here
- `backend_api.py` — the Python↔JS bridge (`Api` class). Reuses the real
  `csv_loader` (+ `minmax_decimate`); returns decimated, JSON-able series. The
  browser never sees the raw 125k-row file. Headless-testable.
- `app_web.py` — pywebview desktop shell that hosts `index.html` and exposes
  `Api` as `window.pywebview.api`.
- `index.html` — minimal real UI (Open CSV → channels → canvas plot + stats) in
  the Claidev2 palette. This is the seam; the full Claude-Design markup folds in
  on top of the same bridge.

## Run
```bash
pip install pywebview            # one-time; uses the OS WebView, no Chromium
python scope_web/app_web.py
```

## Verify the backend half without the GUI
```bash
python scope_web/backend_api.py     # loads a real CSV -> decimated series + stats
```
(PASS proves the risky Python↔data path; the GUI is just a view on top.)

## Why this architecture
- Pixel-faithful to the redesign (the browser renders what Claude Design built).
- Future Claude-Design edits drop in — no Qt re-port.
- Functionality preserved: all compute stays in Python, reusing tested modules.

## Roadmap (next steps, in order)
1. **Fold in the Claidev2 design**: replace `index.html`'s minimal markup with
   the extracted template + styles from `Claidev2/Scope Studio.dc.html`
   (density modes, channel cards, Tools dropdown, collapsible AI rail), keeping
   every control wired to `window.pywebview.api`.
2. **Plot upgrade**: swap the canvas line plot for dual-axis + zoom/pan (uPlot,
   bundled locally) to match the native pyqtgraph interactions.
3. **Wire deterministic tools**: `detect_anomalies`, `estimate_saturation`,
   `reconstruct_rlc`, `column stats` → `Api` methods (no LLM).
4. **Wire the AI chat**: `Api.chat()` → `ai_assistant.ask_model` (MLX backend).
5. **Package**: PyInstaller (pywebview hooks already present in the venv).

The native `app.py` keeps working throughout; this grows beside it until it's
ready to take over.
