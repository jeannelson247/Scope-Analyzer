# Changelog

All notable user-facing changes to Scope Analyzer (formerly Scope Studio). This file is for users and
collaborators; the exhaustive engineering record lives in
`DEVELOPMENT_LOG_SCOPE_STUDIO.txt`. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/).

## [Unreleased]

### Changed
- **Renamed to Scope Analyzer** (was Scope Studio) across the app window title,
  web UI title, and version banner. Repo, history, and prior docs keep the old
  name where it is part of the record.

### Added — web frontend & app packaging (Claude + Codex)
- **Web frontend (Claude-Design):** self-contained, dependency-free UI in a
  native WebView — three-zone layout (controls · workspace · AI rail), density
  modes (Beginner/Standard/Advanced), channel cards, a dual-axis plot with the
  saturation band and dashed RLC-reconstruction overlay, and a collapsible AI
  rail. Opens in a plain browser with a built-in demo for preview.
- **Tested Python↔JS bridge:** a frozen, contract-tested seam — load CSV
  (peak-preserving decimation), per-channel statistics with a caption-ready time
  window, per-channel conversions (gain/offset/formula), preset list,
  deterministic tools (anomaly scan, low-pass), and a persistent calibration
  log. Every call is read-only over the source CSV.
- **CH1 BBCM preset:** H-inverted busbar + 10 kHz low-pass + gain 4, calibrated
  to be comparable to the Pearson channel.
- **Packaging:** three one-file build lanes (Lite / Classic / Mac-MLX) via
  PyInstaller, all sharing one codebase.
- **No-LLM toolbox help** and a regenerable Lite toolbox benchmark pack.
- **Examples dropdown for Lite:** one-click access to all 15 benchmark CSVs,
  with first-run mirroring to `~/Documents/Scope Analyzer/examples/` for normal
  `Open CSV` access and visual QA.

### Added
- Single `version.py` source of truth; the app window title now shows the
  running version so results and screenshots are traceable to a build.
- Closed-form accuracy tests for the math helpers (integrate, gradient,
  low-pass) in addition to the existing engine smoke tests.
- Load-time data-quality checks for rows, sample interval/rate, NaNs,
  nonfinite values, duplicate/backwards timestamps, and large time gaps.
- Structured `<shot>.meta.json` sidecars with source hash, scope metadata,
  QC status, and human-fillable lab fields such as charging voltage and
  module configuration.
- LLM action-boundary tests guarding the "LLM routes; deterministic tools
  compute" invariant.
- Browser-style in-window workspace tabs for 2D plot, Surfaces, Shot data
  3D, GPU 3D, V-I map, and Detail + FFT.
- Benchmark-selected shipping model manifest plus scripts that auto-prefer a
  plugged-in model vault and sync/download only the selected shipping models.
- Real-data RLC reconstruction benchmark for the 6.6 kA `T0000.CSV` shot,
  including before/gap/after fidelity checks for the 5-40 ms reconstruction
  focus interval.
- `docs/METHODS.md` documenting filtering, integration, decimation,
  calibration, RLC reconstruction assumptions, and the local-AI boundary.
- AI annotation trace lines with app version, backend, model field, prompt
  hash, system-prompt hash, max tokens, and source hash; Obsidian exports
  include those traces.

### Fixed
- Lite Examples dropdown now refreshes on click, shows loading/error states,
  searches the macOS app `Contents/Resources` bundle path, and no longer appears
  empty when the Python bridge is late or packaged resources resolve differently.
- Lite help/tool panels now include an explicit `Back to plot` action and a
  capped scroll area so the FAQ cannot push the plot out of view.
- Real-data backtest compatibility with newer NumPy by using
  `np.trapezoid` for charge integration.
- MLX model selection now validates local folders, resolves parent folders to
  complete model children, rejects GGUF files on the MLX backend with a clear
  message, and falls back to the benchmark-selected Qwen3.5-4B default.
- MLX shipping-model sync now detects FAT32 model vaults before downloading
  optional large-shard models, preventing partial downloads on drives that
  cannot store files larger than 4 GiB.

## [0.1.0] — first public preview

### Highlights
- **Three ways to run, one codebase:** PulseLab (no-install web app, no
  LLM), Scope Studio Lite (small desktop build), and Scope Studio Full
  (Mac/MLX desktop build).
- **2D scope view:** robust oscilloscope-CSV loader (Tektronix/Rigol
  preambles, any delimiter), per-channel formula conversions (MATLAB-style),
  forced-through-origin reference calibration with a confidence interval,
  shared low-pass filtering, and publication-quality SVG/PNG/PDF export.
- **3D / analysis views:** analytic surfaces, shot-data 3D, V-I (switching
  locus) map, Detail + FFT, and a GPU cascade — reachable from a single
  **Mode** selector above the plot.
- **Deterministic engine:** all numbers come from NumPy/SciPy; the local LLM
  only routes to tools and explains their output.
- **Local AI side chat:** MLX / Ollama / llama.cpp backends, set
  `SCOPE_STUDIO_LITE=1` to default to a small model on modest machines.
  Optional local-paper RAG (no fine-tuning, no data leaves the machine).
- **Data integrity:** raw CSVs are immutable; the source file is hashed;
  formulas, filters, overlays, and AI reconstructions are reversible
  in-memory state unless you explicitly export a derived file.
- **Quality gates:** headless engine test suite + PulseLab Node backtest,
  both run in CI on every push.

### Known gaps (tracked in docs/PRE_RELEASE_ASSESSMENT.md)
- The model vault must be mounted or selected models must be downloaded before
  MLX direct can run without falling back to a Hugging Face model id.
