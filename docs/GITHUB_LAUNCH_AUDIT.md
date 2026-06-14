# Scope Studio GitHub Launch Audit

Date: 2026-06-14

Purpose: document the actual repository state after the cowork/fable handoff,
apply the five requested expert lenses, and define the remaining work before
publishing to GitHub. UI changes were avoided except where the user explicitly
requested the 2D/3D merge.

## Current Repository State

The repository is initialized on `main`. The current working tree has
uncommitted release-readiness work:

- `CHANGELOG.md` added.
- `version.py` added with `__version__ = "0.1.0"`.
- `tests/test_signal_math.py` added with closed-form accuracy tests.
- `app.py` edited only to include the version in the window title.
- `data_quality.py`, `shot_metadata.py`, and LLM safety/data-quality/metadata
  tests added.
- `scripts/backtest_real_data.py` fixed for newer NumPy.
- `DEVELOPMENT_LOG_SCOPE_STUDIO.txt` updated through Change Set 65.
- `config/shipping_models.json` added with the benchmark-selected MLX
  shipping set.
- `scripts/sync_shipping_models.py` and `scripts/download_mlx_models.sh`
  added/updated so model sync/download targets only the selected shipping
  models and auto-prefers a plugged-in model vault.
- `scripts/backtest_rlc_reconstruction.py` added for the real-data 5-40 ms
  RLC reconstruction benchmark.
- `docs/METHODS.md` added.

The P0 code/tests are now handled but still need to be committed after final
manual UI verification:

- P0 #6: version source of truth + changelog.
- P0 #3: numerical-accuracy tests for `signal_tools`.
- P0 #4: LLM action-boundary tests.
- P0 #2: data-quality report on load.
- P0 #5: structured shot metadata sidecar on load.
- P1 model-vault/model-loader work: handled.
- P1 in-window 2D/3D workspace tabs: handled, offscreen Qt smoke-tested.
- P1 RLC reconstruction benchmark: handled and passing on `T0000.CSV`.
- P1 methods documentation and AI annotation traceability: handled.

The existing GitHub launch helper is:

- `scripts/setup_repo.sh`
- `docs/PUSH_TO_GITHUB.md`

The existing public-readiness assessment is:

- `docs/PRE_RELEASE_ASSESSMENT.md`

## Project Inventory Observed

Core desktop application:

- `app.py`: PySide6 desktop UI, CSV load path, browser-style 2D/3D workspace
  tabs, AI panel, deterministic tools, overlay shots, RLC/saturation overlays.
- `csv_loader.py`: robust oscilloscope CSV/TXT loader with preamble/header/unit
  detection, plus min-max decimation.
- `signal_tools.py`: AST-sandboxed formula helpers: baseline, lowpass,
  integrate, gradient, moving mean, NumPy functions.
- `calibration.py`: forced-origin calibration fit.
- `detect_anomalies.py`: deterministic anomaly scan.
- `rlc_reconstruct.py`: censored maximum-likelihood overdamped RLC
  reconstruction with bootstrap band.
- `saturation_recovery.py`: deterministic saturation/soft-saturation recovery.
- `plot_render.py`: pyqtgraph curve construction; blank-plot regression fixed by
  applying view-dependent downsampling after `addItem`.
- `surface3d.py` and `gl_waterfall.py`: 3D surfaces, shot-data waterfalls,
  V-I maps, GPU cascade support.

AI and safety:

- `ai_assistant.py`: local MLX/Ollama/llama.cpp backends and structured router
  schema.
- `chat_actions.py`: fixed action/tool dispatcher. Unknown tools are rejected;
  numeric tools route to deterministic Python modules.
- `model_catalog.py`: model profiles and `SCOPE_STUDIO_LITE` defaults.
- `tool_sandbox.py` and `tool_sandbox/`: inactive draft-tool workflow.

Testing and benchmarks:

- `tests/test_engine.py`: engine, formula safety, calibration, anomaly,
  decimation, model-catalog tests.
- `tests/test_signal_math.py`: new accuracy tests from Change Set 62.
- `tests/test_llm_action_safety.py`: action-boundary and formula-sandbox
  safety tests.
- `tests/test_data_quality.py`: QC report tests.
- `tests/test_shot_metadata.py`: sidecar provenance/preservation tests.
- `pulselab/backtest_node.js`: web engine backtest.
- `scripts/backtest_real_data.py`: real-shot deterministic backtest.
- `scripts/backtest_rlc_reconstruction.py`: real-data RLC reconstruction
  benchmark for the 5-40 ms focus interval.
- `scripts/benchmark_mlx_models.py`, `scripts/benchmark_reasoning.py`, and
  related reports under `backtests/`.

Docs:

- `README.md`, `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md`.
- `docs/PUSH_TO_GITHUB.md`, `docs/PRE_RELEASE_ASSESSMENT.md`,
  `docs/METHODS.md`,
  `docs/CHOOSE_YOUR_VERSION.md`, `docs/MODEL_SELECTION_GUIDE.md`,
  `docs/USER_GUIDE.md`, `docs/PROJECT_REPORT.md`.

Data observed locally:

- Found: `/Users/jeannelson/Documents/Data Scope/2026-04-20 4 Modules full amperage @ 100% 6.6kA/T0000.CSV`
- Found: `/Users/jeannelson/Documents/Data Scope/2026-04-09 4 Modules in parallel 100% and step current waaveforms/T0012.CSV`
- Existing report: `backtests/real_data_6p6kA_report.txt`
- Created ignored sidecars:
  - `/Users/jeannelson/Documents/Data Scope/2026-04-20 4 Modules full amperage @ 100% 6.6kA/T0000.meta.json`
  - `/Users/jeannelson/Documents/Data Scope/2026-04-09 4 Modules in parallel 100% and step current waaveforms/T0012.meta.json`

The 6.6 kA report shows a real-data benchmark pass:

- 1,250,000 rows
- `CH2` peak about 6554.76 A
- plateau about 6308.35 A
- expected 6600 A, error 4.42%, tolerance 5%

The RLC reconstruction report shows a real-data benchmark pass:

- target `CH1` via `BBCM v2`, reference `CH2`
- Pearson reference trusted 0-5 ms
- censoring lower bound 6000 A
- reconstruction focus 5-40 ms
- before 5 ms NRMSE 2.08%
- after 40 ms NRMSE 4.01%
- gap lower-bound violations 0%

## Five-Lens Assessment

### 1. UI/UX Engineer

What is strong:

- The duplicate plotting controls were consolidated.
- Plot autoscale shortcut is real and scoped to the plot widget.
- 2D plot, 3D surface, shot 3D, GPU 3D, V-I map, and Detail+FFT now share one
  central browser-style workspace tab bar.
- First-run example shot exists.

Remaining risk:

- Manual visual verification is still needed on macOS, but offscreen Qt smoke
  tests instantiate the tabbed workspace and load the bundled example shot.
- Fixed panel sizes can still clip on small screens. This is a usability P1,
  not a launch blocker if README/screenshots set expectations.
- Empty-state guidance is present but not a large canvas overlay. Avoid changing
  UI unless user testing shows confusion.

Recommendation before GitHub:

- Do not touch UI for v0.1 unless a visual smoke test reveals a broken path.
- Add a manual checklist for the user to click: load example, load `T0000`,
  plot CH1/CH2, Mode -> V-I map, Mode -> Shot 3D, anomaly scan.

### 2. Data Analyst

What is strong:

- Loader handles real scope preambles and units.
- Raw files are hashed and kept read-only.
- Overlay shots already exist in 2D.
- Real 6.6 kA backtest exists and passes.

Remaining risk:

- Done in CS64: `data_quality.py` produces a structured QC summary and
  `app.py` shows a one-line load-time QC status.
- There is no user-facing analysis-session artifact that captures selected
  channels, formulas, filters, calibration windows, and visible range.
- Multi-shot voltage cascade requires charging-voltage metadata, but voltage is
  currently not structured.

Recommendation before GitHub:

- Continue by using QC reports in batch/cascade scripts; the core P0 is done.

### 3. Applied Mathematician

What is strong:

- Formula helpers are deterministic and sandboxed.
- New closed-form math tests cover integration, gradient, lowpass, moving mean,
  and baseline.
- RLC reconstruction uses a stated overdamped model and reports bootstrap
  uncertainty.
- Saturation recovery states assumptions and caveats.

Remaining risk:

- FFT accuracy is not yet covered in `tests/test_signal_math.py`.
- Done: RLC reconstruction has a real-data regression for the requested
  5-40 ms reconstruction regime, including checks before and after that
  interval.
- Bootstrap seed is fixed internally (`default_rng(0)`), which is good for
  reproducibility, but the seed is not exposed in report metadata.

Recommendation before GitHub:

- Extend math tests with FFT tone frequency/bin-width checks if FFT helper is
  considered a launch claim.
- Add a real-data RLC reconstruction benchmark using `T0000.CSV`, constrained
  to the 5-40 ms window, and evaluate fidelity outside the fit window.
- Treat reconstruction as an estimate/model overlay, not a measurement, in all
  docs and reports.

### 4. ML Engineer

What is strong:

- LLM does not directly compute arrays; it emits structured actions or text.
- `chat_actions.run_tool` uses a fixed allowlist.
- Formula evaluation is AST-whitelisted and rejects imports/attributes.
- Model benchmark reports already exist.

Remaining risk:

- Done in CS64: `tests/test_llm_action_safety.py` guards malformed JSON,
  router allowlist, unknown-tool rejection, and sandbox rejection of malicious
  formula actions.
- The LLM can set a channel formula. This is acceptable only because formulas
  are sandboxed, visible, undoable, and user-overridable. It should be tested.
- Prompt injection through indexed papers or CSV-derived text should be
  documented as limited-impact: the model may suggest actions, but the engine
  remains the authority and unknown tools do not run.

Recommendation before GitHub:

- Done: AI annotation trace lines include app version, backend, model field,
  prompt hash, system-prompt hash, max tokens, and source hash.

### 5. Scientist Handling Experimental Data

What is strong:

- Raw measurement immutability and SHA-256 hashing are already first-class.
- Obsidian note export includes source hash and tool outputs.
- Real-data benchmark validates the 6.6 kA shot.
- Deterministic tools post results to chat/history for traceability.

Remaining risk:

- Done in CS64: structured shot sidecars are created on load.
- Charging voltage/module configuration/date/operator/scope settings are not
  reliably available for cross-shot analysis.
- Multi-shot cascade by charging voltage cannot be made reproducible until
  metadata exists.
- `T0012.CSV` is present in the April 9 step-current folder and is already
  used by the PulseLab backtest. The broader multi-voltage cascade still needs
  charging-voltage values filled into sidecars.

Recommendation before GitHub:

- Add a headless `shot_metadata.py` module before any UI work.
- Metadata fields should include:
  - `source_path`
  - `source_sha256`
  - `shot_id`
  - `charging_voltage`
  - `module_config`
  - `operator`
  - `scope_model`
  - `sample_interval_s`
  - `notes`
  - `created_at_utc`
- Sidecar filename: `<shot>.meta.json`.
- First implementation can infer `shot_id` from filename and scope/sample data
  from loader metadata, leaving unknown lab fields blank or `null`.

## P0 Status

| P0 | Status | Notes |
|---|---|---|
| Version + CHANGELOG | Done, uncommitted | `version.py`, `CHANGELOG.md`, `app.py` title, log CS62 |
| Math accuracy tests | Done, uncommitted | `tests/test_signal_math.py`, 9 tests |
| LLM never-computes invariant test | Done, uncommitted | `tests/test_llm_action_safety.py` |
| Data-quality report | Done, uncommitted | `data_quality.py`, load-time status line |
| Structured shot metadata sidecar | Done, uncommitted | `shot_metadata.py`, app load hook |
| Mode UX clarity | Accept for v0.1 | No further UI unless visual test fails |

## Recommended Fix Sequence From Here

1. Run manual desktop UI smoke test on macOS.
2. Commit Change Sets 62-64.
3. Add real-data benchmark for `T0000.CSV` reconstruction in the 5-40 ms
   regime.
4. Rerun model benchmark with the reconstruction prompt and select the most
   efficient model that obeys the deterministic-tool boundary.
5. Avoid new UI features for the GitHub launch unless smoke testing exposes a
   blocker.

## User-Data Breach / LLM Attack Review

Threats considered:

- Malicious CSV text or paper excerpt instructs the LLM to leak data.
- Malicious model output tries to run code.
- Malicious model output tries to invent a tool.
- Malicious model output tries to set an unsafe formula.
- Accidental commit of raw measurement data or model weights.

Observed mitigations:

- `SECURITY.md` documents local-only design and raw waveform non-disclosure.
- Raw arrays are not sent to model prompts; tool outputs are derived text.
- `chat_actions.run_tool` rejects unknown tool names.
- `signal_tools` rejects imports, attribute access, and scalar-only formulas.
- `.gitignore` excludes `T0*.CSV`, models, secrets, caches, and backups.
- GitHub setup script prints guard counts before committing.

Remaining launch hardening:

- Done in CS64: action-layer safety tests and a prompt-injection boundary note
  in `SECURITY.md`.
- Add a pre-push checklist command:
  - `git status --short`
  - `git ls-files | grep -E 'T0.*\\.CSV|\\.gguf|\\.env|\\.key|venv/'`
  - `pytest`
  - `node pulselab/backtest_node.js`

## Real-Data Test Plan

### Available now

Run the existing 6.6 kA benchmark:

```bash
python3 scripts/backtest_real_data.py \
  "/Users/jeannelson/Documents/Data Scope/2026-04-20 4 Modules full amperage @ 100% 6.6kA" \
  --expect-amps 6600 \
  --tolerance 0.05 \
  --out backtests/real_data_6p6kA_report.txt
```

### Reconstruction benchmark to add

Use `T0000.CSV` and test reconstruction over 5-40 ms:

- Load `CH2` as the current/reference channel when available.
- Use visible/display time units consistently.
- Fit or reconstruct in 5-40 ms.
- Check pre-window fidelity before 5 ms.
- Check post-window fidelity after 40 ms.
- Report RMS error and peak error against clean/non-censored regions.
- Require deterministic output from fixed seed.

This should be a script under `scripts/` plus a saved report under
`backtests/`, not a UI feature.

### Multi-shot charging-voltage cascade

`T0012.CSV` is present in the April 9 folder. The next blocker is not file
access; it is structured charging-voltage metadata for each shot. Once values
are filled into sidecars:

- Build a manifest of CSV files.
- Create or read `<shot>.meta.json` for charging voltage.
- Sort shots by charging voltage.
- Generate a cascade/surface report: time × charging voltage × current.
- Do not commit raw CSVs; commit only code and anonymized/summary reports.

## GitHub Launch Gate

Do not tag `v0.1.0` until:

- `pytest` passes.
- PulseLab Node backtest passes or cleanly skips missing real data.
- No raw CSV/model/secrets are tracked.
- Change log and development log are current.
- P0 tests for LLM safety, data quality, and metadata sidecar are present.
- Real 6.6 kA benchmark still passes.
- User manually verifies desktop plotting and Mode launcher on macOS.
