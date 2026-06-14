# Scope Studio

Fast oscilloscope-CSV viewer and publication-figure exporter for tokamak
current-driver testing, with formula-based monitor conversion, fit-based
channel calibration, and a local side-chat for quick analysis.

## Which version do I want?

Scope Studio ships three ways from this one repo: a no-install **PulseLab**
web app (no LLM), a small-footprint **Lite** desktop build, and a Mac/MLX
**Full** desktop build. See
[docs/CHOOSE_YOUR_VERSION.md](docs/CHOOSE_YOUR_VERSION.md) for which one to
pick and how to get it.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

No scope file yet? Choose **File ▸ Load example shot** to open a bundled
synthetic capture and try every feature.

For a byte-for-byte reproduction of the tested environment (e.g. to debug a
version-specific issue) install the pinned set instead:

```bash
pip install -r requirements-lock.txt
```

Optional Mac MLX backend (recommended on Apple Silicon):

```bash
pip install -r requirements-mlx-mac.txt
```

### Lite mode (small laptops / student build)

Set `SCOPE_STUDIO_LITE=1` before launching to default the AI side-chat to a
small local model (≤4B; Ollama default `llama3.2:1b`) instead of the
heavyweight M4-Pro default. You can still pick any installed model in the
AI panel.

```bash
SCOPE_STUDIO_LITE=1 python app.py
```

Optional llama.cpp fallback:

```bash
pip install llama-cpp-python
```

## What is new in this version

- Formula presets are no longer limited to gain/offset. Each channel can
  use expressions such as:

```text
(x - 2.5) * 200 / 2
baseline((x - 2.5) * 200 / 2, t_ms, -1.0)
baseline(integrate(lowpass(x, t, 1e4), t, negate=True), t_ms, -1.0)
movmean(x, samples(5e-3))
```

- Built-in presets now cover the MATLAB-style busbar workflows, Pearson
  monitors, Rogowski/B-dot style signals, and HV probes.
- The **Formula editor / save preset...** button opens a larger formula
  text box for the selected channel, validates the expression, and can
  save it as a reusable preset in `presets.json`.
- A **Reference calibration** panel applies the same forced-through-origin
  fit you use in MATLAB: choose a source channel, choose a reference
  channel, fit over the visible or specified X window, and the fitted
  slope is multiplied into the source-channel gain.
- The plot palette can be switched between **Wong / Nature**,
  **Nature Physics (NPG)**, and a **Tokamak dual-axis** palette.
- The right-side **Local AI side chat** can summarize the visible data or
  answer custom questions about the currently displayed traces.
- A **Signal processing** panel can apply a low-pass cutoff to all traces,
  left-axis traces, or current-like traces before plotting/statistics/export.
- The side chat can run a deterministic anomaly scan for spikes, clipping,
  drift, crest factor, dropouts, and module imbalance.
- A folder of papers can be indexed for retrieval. This is **RAG**, not
  fine-tuning: the app searches local paper excerpts and adds the most
  relevant ones to the prompt at question time.
- Raw CSV files are treated as immutable measurement inputs. Scope Studio
  records a source-file hash when loaded; formulas, filters, AI overlays,
  reconstructions, interpolation, and extrapolation are display/session
  state in RAM unless you explicitly export a new derived file.
- The AI panel now has an **Undo AI change** button and a **Reset display**
  button. Reversible actions affect the displayed view, not the original CSV.
- The 3D window includes synthetic V/I/di-dt examples using `L = 166 uH`,
  plus direct loading in the **Shot data 3D** tab.
- The experimental tool sandbox can create inactive draft-tool templates.
  Draft tools are not imported or approved until reviewed and tested.

## Quick start

1. Open a CSV or TXT scope export.
2. Pick the time column in **X column**. The default `X × = 1000` shows
   seconds as milliseconds.
3. In **Channels**, tick what you want to plot.
4. Choose a **Preset** for each monitor. The **Formula** column is fully
   editable if you want to mirror a custom MATLAB conversion.
5. Use **Reference calibration** if you want to fit a busbar or Rogowski
   signal against a Pearson reference over the rise window.
6. Enable **Signal processing** if you want a shared low-pass cutoff.
7. Use the control row above the plot for pan / box zoom / X-only /
   Y-only navigation and zoom in/out. Click the plot and press **A** to
   autoscale (reset the view).
8. Export SVG / JPG / PNG / PDF from the current view using the
   publication renderer.

## Synthetic 3D V/I/di-dt examples

Generate the educational 166 uH examples with:

```bash
python scripts/generate_synthetic_vi_didt.py
```

This writes:

- `examples/synthetic_vi_didt_scope.csv`
- `examples/synthetic_vi_didt_surface_current.csv`
- `examples/synthetic_vi_didt_surface_didt.csv`
- `examples/synthetic_vi_didt_surface_voltage.csv`
- `examples/synthetic_vi_didt_README.txt`

The model is deterministic and intentionally simple:

```text
L = 166e-6 H
dI/dt = (V_drive(t) - R*I) / L
L*dI/dt = inductive voltage estimate
```

Use the scope-style file in the main plotter or **Shot data 3D**. Use the
surface files in **3D surface view -> Surfaces**. These are demonstration
datasets for testing plotting and analysis workflows, not experimental data.

## Formula reference

The formula sandbox exposes:

- `x`: the raw selected Y column
- `t`: the selected X column in seconds
- `t_ms`: the selected X column in milliseconds
- `baseline(y, axis, end[, start])`
- `lowpass(y, t, cutoff_hz)`
- `integrate(y, t, negate=False)`
- `movmean(y, window)`
- `samples(seconds[, minimum])`: converts a time window, such as `5e-3`,
  into the correct number of samples for the loaded CSV
- `gradient(y, axis)`
- `abs`, `clip`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `where`

Gain and offset are still applied **after** the formula.

## MATLAB-derived calibration presets

These fixed conversions from the MATLAB scripts are now available in the
preset menu:

- Busbar CH1 750 A/V with baseline and 15 kHz low-pass:
  `baseline(lowpass((x - 2.5) * 1500 / 2, t, 1.5e4), t_ms, -1.0)`
- Pearson CH2 direct current reference with baseline and 15 kHz low-pass:
  `baseline(lowpass(x, t, 1.5e4), t_ms, -1.0)`
- Rogowski CH3 raw voltage with baseline and 15 kHz low-pass:
  `baseline(lowpass(x, t, 1.5e4), t_ms, -1.0)`
- Busbar CH1 100 A/V with baseline and 10 kHz low-pass:
  `baseline(lowpass((x - 2.5) * 200 / 2, t, 1e4), t_ms, -1.0)`
- OP AMP / CH4 raw voltage with baseline and 10 kHz low-pass:
  `baseline(lowpass(x, t, 1e4), t_ms, -1.0)`
- Busbar polarity form from `(2.5 - V)`: `(2.5 - x) * -750`
- Small-channel form from `(2.5 - V)`: `(2.5 - x) * -100`
- Large loop channel scale: `x * (1500 / 4)`
- Precomputed voltage-difference current: `x / 0.08`
- Moving-average helpers: `movmean(x, samples(5e-3))` and
  `movmean(x, samples(10e-3))`

The forced-through-origin calibration ratios from MATLAB,
`ratio = dot(source, reference) / dot(source, source)`, are computed by
the **Reference calibration** panel because they depend on the selected
shot, reference channel, and fit window.

## Local AI and papers

- `MLX direct`: preferred Mac path; install `requirements-mlx-mac.txt` and
  use an MLX/Hugging Face model name such as
  `mlx-community/Qwen3.5-4B-MLX-4bit`. The app auto-scans a plugged-in
  model vault at `/Volumes/<drive>/ScopeStudioModels/mlx`,
  `/Volumes/<drive>/models/mlx`, or `/Volumes/<drive>/mlx`, then falls back
  to `~/models/mlx`. To fetch only the benchmark-selected shipping set, run
  `./scripts/download_mlx_models.sh`. GGUF files are for llama.cpp, not
  direct MLX.
- `Ollama`: switch backend and use a model name such as `qwen2.5:7b`.
- `llama.cpp`: point the model field to a `.gguf` file.
- Papers folder: PDF, TXT, Markdown, TeX, and MATLAB `.m` files can be
  indexed. PDF extraction uses `pypdf`.

The local model sees:

- stats from the visible window
- a small decimated waveform preview for each enabled trace
- relevant paper excerpts, if indexed

It does **not** receive the full raw CSV by default.

The assistant may request approved deterministic tools and can help draft
new tools in `tool_sandbox/drafts/`, but draft tools are inactive until
reviewed and promoted. The AI must not overwrite source CSVs or silently
change calibration constants.

## Tests

The deterministic NumPy engine (CSV parsing, formula sandbox, calibration,
anomaly detection, decimation, model selection) has a headless test suite —
no Qt or GPU needed:

```bash
source venv/bin/activate                       # use the project venv, not system Python
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -q
```

If `pytest` reports "No module named pytest", you're on the wrong Python —
activate the venv first (the line above), then reinstall.

The PulseLab web engine has its own Node backtest:

```bash
node pulselab/backtest_node.js                 # engine-only checks
node pulselab/backtest_node.js "/path/to/Data Scope"   # + real-shot benchmarks
```

Both run in CI on every push (`.github/workflows/tests.yml`).

## Notes

- The export styling now follows the MATLAB Nature template more closely:
  thinner axes, minor ticks, legend without a box, and Nature-friendly
  colour defaults.
- If a channel formula is invalid, the plot skips that trace and shows the
  reason in the status bar instead of crashing the app.
