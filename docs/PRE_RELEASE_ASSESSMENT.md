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

## P0/P1 status after Change Set 65

- **Done:** version source of truth + CHANGELOG.
- **Done:** numerical-accuracy tests for math helpers.
- **Done:** LLM action-boundary safety tests.
- **Done:** load-time data-quality report.
- **Done:** structured shot metadata sidecar.
- **Done:** 2D and 3D analysis views now share one browser-style workspace
  tab bar in the main window.
- **Done:** benchmark-selected MLX shipping model manifest and model-vault
  sync/download scripts.
- **Done:** real-data RLC reconstruction benchmark for the 5-40 ms focus
  interval on `T0000.CSV`.
- **Done:** `docs/METHODS.md` and AI annotation trace metadata.

---

## 1. UI/UX engineer

**Strong:** single consolidated navigation control (Mode + pan/zoom),
colorblind-safe Wong palette, first-run "Load example shot", honest `A`
autoscale shortcut, formula errors surfaced in the status bar instead of
crashing.

**Gaps**
- **P0 — Done in CS65.** 2D plot, Surfaces, Shot data 3D, GPU 3D, V-I map,
  and Detail + FFT now live in a single central workspace tab bar.
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
- **P0 — Done in CS64.** A one-line QC banner now reports rows, detected
  sample rate, NaN/nonfinite count, duplicate/backwards timestamp checks,
  and large timestamp gaps on load.
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
- **P0 — Done in CS62.** Closed-form tests now cover `integrate`,
  `gradient`, `lowpass`, `movmean`, and `baseline`. FFT-specific accuracy
  can remain a P1 if the FFT view becomes a publication claim.
- **P1 — Done in CS65.** The RLC real-data benchmark
  (`scripts/backtest_rlc_reconstruction.py`) reports before/gap/after
  fidelity for the 6.6 kA `T0000.CSV` shot and writes
  `backtests/rlc_reconstruction_6p6kA_report.txt`.
- **P1 — Done in CS65.** `docs/METHODS.md` documents filter type and edge
  handling, integration rule, decimation guarantees, calibration model, RLC
  equations, censoring assumptions, and AI/tool boundaries.
- **P2 — Filter edge effects / phase.** Note whether low-pass is
  zero-phase (filtfilt) or causal; document the trade-off for timing-
  sensitive edge analysis.

## 4. ML engineer

**Strong:** clean separation — the LLM never computes numbers (enforced by
design: actions route to NumPy tools); multiple backends with graceful
fallback; a real benchmark harness; RAG over local papers instead of
fine-tuning; local-only / no telemetry.

**Gaps**
- **P0 — Done in CS64.** `tests/test_llm_action_safety.py` guards malformed
  JSON handling, fixed router allowlist, unknown-tool rejection, and formula
  sandbox rejection of malicious model-suggested formulas.
- **P1 — Done in CS65.** AI replies now log app version, backend, model
  field, prompt hash, system-prompt hash, max tokens, and source hash; the
  Obsidian export includes the trace metadata.
- **P1 — Done in CS65.** `config/shipping_models.json` pins the
  benchmark-selected shipping model set, while `scripts/download_mlx_models.sh`
  and `scripts/sync_shipping_models.py` auto-prefer a plugged-in model vault
  and avoid copying the exploratory benchmark fleet.
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
- **P0 — Done in CS64.** A structured `<shot>.meta.json` sidecar is created
  on load, containing source hash, scope model, sample interval, row/column
  metadata, QC status, and human-fillable lab fields including charging
  voltage and module configuration.
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

1. **Mode UX:** done as in-window workspace tabs; visual smoke test still
   required on macOS before tagging.
2. **Data-quality banner on load:** done.
3. **Numerical-accuracy tests:** done for core helpers; FFT-specific test is
   P1.
4. **LLM never-computes invariant test:** done.
5. **Structured shot metadata sidecar:** done.
6. **`__version__` + CHANGELOG:** done.

Everything else (P1/P2) is a fast-follow. None of the P0s are large; most
are a few hours each and several are pure-NumPy/headless, so they fit the
existing test-gated workflow without new infrastructure.

## Suggested sequence

Next sequence: run the manual macOS UI smoke test, mount the model vault or
download the selected shipping models, then continue into P2/P3 items such as
session save/restore, batch/cascade metadata enrichment, and packaging polish.
