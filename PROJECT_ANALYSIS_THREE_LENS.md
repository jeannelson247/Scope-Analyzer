# Scope Studio — Three-Lens Project Analysis

Written 2026-06-20. Companion to `SCOPE_STUDIO_HANDOFF.md`. Read through the three
project lenses: **Software Engineer**, **Data Analyst**, **Scientist who wants to
streamline data analysis and interpretation**. Grounded in the current `scope_studio03`
tree (51 passing tests; `app.py` ~2,990 LOC; web bridge in `scope_web/`).

---

## 0. Snapshot

| Signal | State |
|---|---|
| Native app (`app.py`, PySide6) | Works; 3-pane `QSplitter` layout |
| Web frontend (`scope_web/`) | Bridge foundation built + headless-verified; UI fold-in not started |
| Tests | 51 passing, incl. LLM action-boundary + FFT/signal-math accuracy |
| Docs | `DEVELOPMENT_LOG…` (160 KB), `CHANGELOG.md`, `docs/METHODS.md`, handoff |
| Determinism boundary | Enforced + tested ("LLM routes; deterministic tools compute") |
| Git | Many high-value files untracked; large `.zip` design bundles in tree |

The bones are unusually strong for a solo scientific tool. The risk is not capability —
it's **two parallel front-ends, untracked work, and a widening surface that one person
maintains.** Each lens below sharpens that.

---

## 1. Software Engineer lens

### What's working
- **Clean backend/UI separation.** `backend_api.py` reuses the *same* tested
  `csv_loader` / `minmax_decimate` as the native app and is importable without
  pywebview, so the data path is unit-testable headlessly. This is the right seam.
- **Determinism is a tested invariant, not a hope.** `test_llm_action_safety.py`
  guards "LLM routes; deterministic tools compute." That is the single most important
  architectural property of this app and it has a test wall around it.
- **Real engineering hygiene:** single `version.py` source of truth surfaced in the
  window title, `requirements-lock.txt`, `pytest.ini`, closed-form accuracy tests,
  load-time data-quality checks, `.meta.json` sidecars with source hashes.

### Risks / debt (highest first)
1. **Untracked, irreplaceable work.** `git status` shows `app.py`, `chat_actions.py`,
   `rlc_reconstruct.py`, `scope_web/`, and the dev log all modified-or-untracked, plus
   three multi-MB design `.zip`s sitting in the working tree. One bad `git clean` and
   the web bridge + latest RLC work is gone. **Commit `scope_web/` and the source diffs
   now; move `.zip`/`Claidev2/`/`claude_design_upload/` out of the repo or into
   `.gitignore` + a `design/` archive.**
2. **`app.py` is a 2,990-line `MainWindow` god-class** (~114 methods). It mixes
   layout, plotting, channel state, formula eval, dialogs, and worker threads. Before
   the web migration eats your attention, extract the non-Qt logic (channel model,
   formula validation, recovery-settings assembly) into plain modules the *web* bridge
   can also call. Otherwise you reimplement it in JS.
3. **Two front-ends, one developer.** The native app and `scope_web/` will drift. Decide
   explicitly: is native now in **maintenance freeze** (bug-fix only) while web becomes
   primary? The handoff implies yes — make it a written rule so you stop paying double.
4. **Bridge surface is stubbed silently-safe but incomplete.** `chat()` and
   `list_models()` return honest "not wired" payloads (good), but there's no schema/
   contract test on the bridge yet. Add a `test_backend_api.py` that asserts the
   `load_csv` return shape, so the JS side has a frozen contract before you build UI on it.

### SWE next 3 moves
- `git add scope_web/ && commit`; relocate design zips; one commit per logical change.
- Add `tests/test_backend_api.py` (return-shape contract + decimation invariant).
- Extract `channel.py` / `recovery_settings.py` pure-logic modules from `app.py`.

---

## 2. Data Analyst lens

### What's working
- **Never mutates the source CSV** — all transforms live in session state. This is the
  cardinal rule of trustworthy analysis and it's baked in.
- **Decimation is honest.** `minmax_decimate` to ~4,000 points preserves peaks/troughs
  (min-max envelope), so a 125k-row capture renders fast *without hiding the spikes that
  matter* — exactly the failure mode naive every-Nth decimation creates.
- **Data quality is checked at load:** sample interval/rate, NaNs, nonfinite, duplicate/
  backwards timestamps, large gaps. An analyst is warned before trusting the trace.

### Gaps
1. **Provenance is strong; reproducibility-of-a-session is weaker.** `.meta.json`
   sidecars hash the *source*, but there's no single "analysis recipe" artifact that
   records: which channels, which formulas, which transforms, which fit windows produced
   *this* figure. For a data analyst, the export should round-trip — a saved figure
   should carry (or link) the exact steps to regenerate it.
2. **`column_stats` is thin.** min/max/mean/std only. Analysts will immediately want
   median/percentiles, RMS, sample count after NaN-drop, and the time-window the stats
   were computed over (full vs. visible range). Cheap to add, high daily value.
3. **No explicit units/dimensionality guard in the bridge.** `units` are passed through
   as strings; nothing stops a formula from adding volts to amps. A light unit-tag check
   would catch real analyst mistakes.

### Data-Analyst next 3 moves
- Extend `column_stats`: median, p5/p95, RMS, N-finite, and the window used.
- Define a `session.json` "analysis recipe" that exports with every figure and reloads.
- Surface the load-time QC verdict in the web UI (a colored chip), not just in logs.

---

## 3. Scientist lens (streamline analysis + interpretation)

### What's working — and it's the crown jewel
- **Censored-ML RLC reconstruction is genuinely good science.** Treating clipped samples
  as *lower bounds* (Tobit-style hinge) rather than real readings is the correct
  statistical move, and it's deterministic NumPy/SciPy with a residual-bootstrap 95%
  band. Plain OLS would bias the peak low; this doesn't. The recent `trusted_windows`
  work lets the scientist separate fit window / saturated interval / trustworthy-sensor
  regions — that's the right set of knobs, and the report line makes the assumption
  **auditable**.
- **The AI never computes the number.** For a publishing scientist this is the difference
  between a tool you can cite and one you can't. The boundary is documented in
  `METHODS.md` and the annotation trace records app version, model, prompt hash,
  system-prompt hash, and source hash. That trace is publication-grade.

### What would streamline interpretation most
1. **Show the model its evidence before it fits.** The dev log already flags this:
   shade trusted windows and censored windows *on the plot* before running the
   reconstruction. Scientists trust a fit they can see the inputs of. This is the single
   highest-leverage interpretive upgrade.
2. **Make the uncertainty band a first-class citizen.** You compute a 95% CI on the peak
   — surface "true peak = X kA (95% CI a–b)" as a copyable result string, not just a
   curve. That sentence is what goes in a figure caption.
3. **Close the loop from interpretation to record.** The Obsidian/`METHODS` plumbing
   exists; a one-click "log this shot's reconstructed peak + CI + assumptions to the
   shot's `.meta.json` and notes" would turn analysis into a durable lab record — which
   is the stated project goal ("streamline data analysis and interpretation").
4. **Methods text should travel with the result.** When a scientist exports the RLC
   overlay, attach the 2–3 sentence METHODS description of censored-ML so a co-author or
   reviewer reads *why* the plateau doesn't bias the peak.

### Scientist next 3 moves
- Pre-fit plot shading of trusted vs. censored windows.
- Peak-with-CI as a copyable caption string on every reconstruction.
- One-click "write result + assumptions back to shot record."

---

## 4. Cross-cutting priority list (do in this order)

1. **Commit and de-risk the tree** (SWE #1) — protect the work that already exists.
2. **Freeze native, commit to web as primary** — stop the two-front-end tax.
3. **Bridge contract test + extract pure logic from `app.py`** — so the web build reuses
   Python instead of reimplementing it in JS.
4. **Pre-fit window shading + peak-with-CI string** — the two changes that most improve
   day-to-day scientific trust.
5. **Richer stats + session-recipe export** — analyst reproducibility.

The through-line across all three lenses: **the science and the safety boundary are
already excellent; the active risk is operational** (untracked work, duplicated
front-ends, a monolithic UI file). Spend the next sessions hardening the seam you've
already built rather than adding new tools.
