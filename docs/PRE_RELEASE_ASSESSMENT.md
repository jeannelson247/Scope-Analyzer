# Scope Studio — Pre-Release Assessment

What's solid and what's missing before a public v0.1, read through five
professional lenses. Items are tagged **P0** (block release), **P1** (do
soon after), **P2** (nice to have). This is an honest gap list, not a
victory lap — the foundation is strong; the gaps are mostly about trust,
reproducibility, and first-contact polish.

Reference environment: macOS, Apple Silicon (M4 Pro), Python 3.14. The
deterministic engine is the source of all numbers; the local LLM only
routes and explains (never computes).

---

## Snapshot — what already exists

- Three deliverables from one repo: **PulseLab** (no-install web app, no
  LLM), **Lite** desktop, **Full** desktop (Mac/MLX). Build + Pages CI in
  place (`.github/workflows/`).
- Deterministic engine: robust CSV loader, formula sandbox (AST-validated),
  forced-origin calibration, anomaly detection, peak-preserving decimation,
  RLC reconstruction, saturation estimate.
- Headless test suite (17 tests) + Node backtest, both gated in CI.
- Provenance: raw CSVs treated as immutable; SHA-256 source hashing;
  reversible in-RAM transforms; export of derived copies only.
- Local LLM: MLX / Ollama / llama.cpp backends, two-tier router+interpreter,
  `SCOPE_STUDIO_LITE` for small machines.
- LICENSE, CONTRIBUTING, SECURITY, README quickstart, change log.

---

## 1. UI/UX engineer

**Strong:** single consolidated navigation control (Mode + pan/zoom),
colorblind-safe Wong palette, first-run "Load example shot", honest `A`
autoscale shortcut, formula errors surfaced in the status bar instead of
crashing.

**Gaps**
- **P0 — Mode model is half-unified.** The Mode dropdown *launches* the 3D
  family in a separate window. For a beginner this is two mental models. At
  minimum, label it so expectations are set (done); ideally embed 3D/V-I as
  real in-window panes so "Mode" truly switches the central view.
- **P1 — No persistent empty/onboarding state on the canvas.** The plot is
  blank until a file loads; the guidance lives only in a small label. A
  centered "Drop a CSV here / Load example" overlay would de-risk first
  contact.
- **P1 — Fixed splitter sizes (500/880/420 px).** On a 13" screen or when
  the window is small, panels clip (your screenshots show the AI panel
  cut off). Needs min-width + scroll, or a collapsible side panel.
- **P2 — No light/dark theme toggle** though `style.qss` exists; follow the
  OS appearance.
- **P2 — Transient errors only.** A small, dismissible log/notification
  area would beat status-bar messages that vanish in 9 s.
- **P2 — App identity:** no icon / bundle art; the title bar still reads a
  generic name. Matters for a "real product" feel at release.

## 2. Data analyst

**Strong:** immutable raw data + hashing, V→A presets, multi-format figure
export (SVG/PNG/PDF), visible-window statistics, overlay shots.

**Gaps**
- **P0 — No data-quality summary on load.** A one-line QC banner (rows,
  detected sample rate, NaN/dropout count, duplicate-timestamp check,
  monotonic-time check) would catch bad files before analysis, not after.
- **P1 — Reproducible analysis sessions.** Presets persist, but there's no
  single "save/restore this whole analysis" (channels, formulas, filters,
  window, calibration) artifact a colleague can reopen. `data_session.py`
  is a start — promote it to a user-facing Save/Open Session.
- **P1 — Only CSV/TXT input.** The PXI/NI roadmap implies HDF5/TDMS and
  `.npz`; a pluggable loader interface would future-proof this.
- **P2 — No batch mode.** Analysts will want to run the same pipeline over a
  folder of shots headlessly (CLI), not click each one.
- **P2 — Units are cosmetic** (axis labels), not enforced/propagated through
  formulas; an accidental V-as-A mislabel won't be flagged.

## 3. Applied mathematician

**Strong:** peak-preserving min-max decimation (display-only, never alters
exported data), forced-origin gain fit *with a confidence interval*, Hann
window on the FFT, censored-ML RLC fit that states its overdamped
assumption and reports a bootstrap CI.

**Gaps**
- **P0 — Numerical methods are under-tested.** The suite covers parsing,
  formula safety, calibration, decimation, anomalies — but **not** the math
  helpers: low-pass response, `integrate` accuracy vs analytic, `gradient`,
  FFT amplitude/peak-frequency. Add accuracy tests against closed-form
  signals (e.g. integrate a sine, recover −cos; FFT a tone, recover its
  frequency within bin width).
- **P1 — Determinism of stochastic paths.** The RLC bootstrap CI should
  expose a seed so a given shot yields a repeatable interval (reproducibility
  is a publication requirement).
- **P1 — Document the methods.** A short `docs/METHODS.md`: filter type and
  edge handling, integration rule (trapezoid?), decimation guarantees,
  calibration model and its CI assumptions, RLC model equations and validity
  domain. This is what a reviewer or co-author will ask for.
- **P2 — Filter edge effects / phase.** Note whether low-pass is
  zero-phase (filtfilt) or causal; document the trade-off for timing-
  sensitive edge analysis.

## 4. ML engineer

**Strong:** clean separation — the LLM never computes numbers (enforced by
design: actions route to NumPy tools); multiple backends with graceful
fallback; a real benchmark harness; RAG over local papers instead of
fine-tuning; local-only / no telemetry.

**Gaps**
- **P0 — The "LLM never computes" invariant isn't guarded by a test.** It's
  the project's central safety claim. Add a test asserting the action schema
  exposes no free-form arithmetic and that numeric fields are tool outputs,
  so a future refactor can't quietly violate it.
- **P1 — Reproducibility of AI output.** Log model name + version + prompt
  hash + decoding params alongside any AI annotation, so an explanation can
  be traced. Pin model identifiers (you pin Python deps; do the same for
  models).
- **P1 — First-run model acquisition.** No model = degraded experience. A
  guided "no model found → here's how to pull one" flow (the roadmap's
  first-run downloader) is needed for non-expert installs.
- **P2 — Eval in CI (smoke).** A tiny offline routing eval (no model
  download) that checks the action-router prompt/JSON contract hasn't
  regressed.
- **P2 — Prompt-injection surface.** Indexed papers/CSV text flow into
  prompts; document and constrain (the engine is the authority, so impact is
  limited, but worth a note in SECURITY).

## 5. Scientist handling experimental data

**Strong:** raw data immutability + hashing is exactly right for lab work;
Obsidian session-note export; calibration mirrors the lab's MATLAB ratios;
deterministic, inspectable tools.

**Gaps**
- **P0 — No structured shot metadata.** Charging voltage, module config
  (S1–S4), date, operator, scope settings live only in filenames/memory.
  A sidecar (`<shot>.meta.json`) captured on load is the prerequisite for
  the multi-shot waterfall ("depth = charging voltage") and for any
  cross-shot comparison or archival.
- **P1 — Citable methodology + versioning.** Add `__version__` and a
  user-facing CHANGELOG so a result can be tied to a software version in a
  lab notebook or paper.
- **P1 — Archival export.** A "save analyzed shot + metadata + method
  parameters" bundle (CSV/NPZ + JSON sidecar) so a processed result is
  self-describing years later.
- **P2 — Experiment log.** Optionally link shots to conditions in a small
  index (ties into the lab_memory/journal pieces already present).

---

## Minimum bar for a credible public v0.1 (the P0s)

1. **Mode UX:** ensure the 2D/3D/V-I/anomaly switch is unambiguous (label or
   embed) — no dead ends for a first-time user.
2. **Data-quality banner on load** (rows, sample rate, NaN/dropout, time
   monotonicity).
3. **Numerical-accuracy tests** for the math helpers (integrate, gradient,
   low-pass, FFT) against closed-form signals.
4. **A test guarding the "LLM never computes numbers" invariant.**
5. **Structured shot metadata sidecar** captured on load.
6. **`__version__` + CHANGELOG** so results are traceable to a build.

Everything else (P1/P2) is a fast-follow. None of the P0s are large; most
are a few hours each and several are pure-NumPy/headless, so they fit the
existing test-gated workflow without new infrastructure.

## Suggested sequence

P0 #6 and #2 first (cheap, high trust), then #3 and #4 (pure tests, protect
the math and the safety claim), then #5 (unlocks multi-shot), then #1 (the
only one touching GUI heavily). Tag `v0.1.0` after #1–#6 land green in CI.
