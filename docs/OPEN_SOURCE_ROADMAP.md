# Scope Studio 03 Roadmap

Scope Studio aims to become an AI-assisted plotting and analysis tool for
students and researchers who work with experimental data but find plotting
standards tedious or difficult to implement consistently.

## Phase 1 - Mac-first scientific plotting assistant

- Fast oscilloscope CSV loading.
- Journal-style plot presets.
- Formula-based current monitor calibration.
- Local AI side chat with MLX-first Mac backend and Ollama/llama.cpp fallbacks.
- Whitelisted AI plot-control actions.
- Immutable raw-data policy: source CSVs are read-only; transforms and
  overlays are in-memory display state unless explicitly exported.
- Reversible AI/display actions with an undo stack.
- Synthetic 166 uH V/I/di-dt examples for 3D plotter validation.
- Local preference/adaptation logging.
- Model selection guide for lightweight and heavyweight profiles.

## Phase 2 - Better measurement adaptability

- Instrument profiles for Tektronix, Keysight, Rigol, Lecroy, and generic CSV.
- Scope-setting detection from headers.
- Multi-channel formula support, such as `CH1 - CH2`.
- Named lab templates for channel mappings, sensors, filters, and export style.
- Per-device model benchmark recommendations.
- Draft-tool sandbox where the assistant can propose tools that remain
  inactive until reviewed and tested.
- Journal-figure style and digitized-curve comparison helpers.

## Phase 3 - Cross-platform packaging

- macOS app bundle.
- Windows installer.
- Linux AppImage or Flatpak.
- Optional model downloader.
- Offline documentation.

## Phase 4 - International educational release

- Translation framework.
- Student tutorials.
- Example datasets.
- Reproducible figure templates.
- Community-submitted journal presets and instrument profiles.

## AI Safety Boundary

The open-source release should keep AI actions whitelisted. The LLM may adjust
plot settings, formulas, styles, filters, and labels through explicit actions,
but it should not silently rewrite application source code or modify original
CSV files. Draft tools must stay sandboxed until promoted by a maintainer.
