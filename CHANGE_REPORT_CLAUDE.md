# Scope Studio — Change Report (Claude, this session)

Sequential record of all changes, why each was made, and how it aids the
program. Continues from CHANGE_REPORT_SEQUENTIAL.txt (Codex change sets
01–02). Tested headlessly against T0001.CSV (Tektronix DPO2024B,
1,250,000 rows).

============================================================
## Change Set 03 — NEW FILE: detect_anomalies.py
Deterministic anomaly detection for the visible window.

**Why this file exists**
- The organizer/side-chat design rule is: *the LLM never computes numbers*.
  A 4–9B local model routes tasks and interprets results; NumPy does the
  math. This module is the math.
- It implements the checks you described: averages/stats already existed;
  this adds spikes, clipping, drift, crest factor, and S1…S4 module
  imbalance, each returning exact numbers the model can then explain.

**Logic, detector by detector**
- `analyze_channel()` — spike detection: subtract a moving mean (window
  ≈ n/2000 samples) to isolate fast structure, then score the residual with
  a **robust z (MAD-based σ)**. MAD instead of std because your spikes are
  exactly what would inflate a standard deviation and hide themselves.
  Events are grouped (`_group_events`) so one ringing burst counts once.
- `_ringing_freq()` — zero-crossing count around each event estimates the
  oscillation frequency. This is what distinguishes EMP pickup from real
  current: in your slides the bus-bar monitor showed ~138 kHz ringing the
  Pearson didn't see. On T0001 it correctly reports CH2 events ringing at
  ~769 cycles/ms (≈769 kHz).
- **Periodic-event guard** — first test on T0001 listed CH1's PWM edges as
  301 "anomalies". A square control wave triggers any spike detector at
  every edge. Fix: if ≥20 events occur and their spacing histogram
  collapses onto ≤4 discrete values covering ≥80% of gaps (this handles
  asymmetric duty cycles, where naive std/mean fails), report them as one
  line: "periodic transition events … consistent with PWM/switching
  edges". Verified: random synthetic spikes are NOT misclassified.
- Clipping: long flat runs pinned at the global extremes (scope
  saturation signature).
- Baseline drift: linear fit over the first 10% of the window, flagged
  when the total drift exceeds 5σ of the residual noise and 1% of range.
- Crest factor: peak/RMS > 5 flags spike-dominated channels (your
  peak ≫ RMS heuristic, now quantified).
- `detect()` cross-channel: among channels labeled as currents
  ("(A)"/"current"), compares peaks and flags >10% deviation from the
  mean — the module-balance check for parallel S1…S4 drivers.
- All thresholds are arguments so the chat can tune them
  (`threshold_sigma`, `crest_limit`, `imbalance_limit`).

**How it aids the program**: anomaly answers become reproducible and
instant (no LLM latency for the scan itself), and small local models stop
being a bottleneck because they only interpret, never calculate.

============================================================
## Change Set 04 — NEW FILE: chat_actions.py
JSON action layer: the model can format plots and run tools.

**Why this file exists**
- You asked for (a) the model formatting plots from examples while keeping
  the manual boxes authoritative, and (b) an organizer that dispatches
  tasks ("analyze", "average", "detect anomalies") to scripts.
- Both are the same mechanism: the model ends a reply with one fenced
  ```json {"actions":[…]} ``` block; the app parses and executes it.

**Logic**
- `ACTION_SCHEMA` — appended to the system prompt; defines formatting
  actions (title, axis labels, x/y ranges, x-scale, zero alignment, top
  axis, per-channel enable/axis/gain/offset/label/**formula**) and tool
  actions (`compute_stats`, `detect_anomalies` with thresholds). It
  instructs the model to emit the block only when asked, never to invent
  calibration numbers, and to interpret (not recompute) tool output.
- `apply_actions()` — formatting actions literally type into the same
  widgets you use (`ed_title.setText` + `editingFinished.emit()`), so the
  manual boxes stay the single source of truth and you can override
  anything by hand afterwards. Channel edits clear `_transform_cache` and
  rebuild the table so formulas re-evaluate.
- `run_tool()` — extracts the visible window once (`_visible_arrays`,
  honoring the current zoom and channel formulas) and dispatches to
  `detect_anomalies.detect()` or `win.compute_stats()`. Tool output is
  returned as text for the chat.
- Defensive everywhere: every widget access guarded, every action wrapped
  in try/except, failures reported as chat lines instead of crashing.

**How it aids the program**: one protocol gives you AI plot formatting,
AI-triggered calibration formulas, and the task-organizer dispatch — all
without giving the model any ability to execute arbitrary code (it can
only name whitelisted actions; formulas still pass through
signal_tools' AST sandbox).

============================================================
## Change Set 05 — app.py patches (5 small, surgical edits)

1. **CHAT_SYSTEM_PROMPT += ACTION_SCHEMA** (after the existing prompt
   definition). Why: the model can't use a protocol it hasn't been told
   about. Kept as a separate concatenation so the original prompt text is
   untouched and the schema lives in one file (chat_actions.py).

2. **`_ai_done()` rewritten to route through `process_reply()`.**
   Why/logic: replies are split into (clean text → shown as Assistant),
   (formatting changes → shown as a "System: Applied: …" line so you
   always see what the AI touched), and (tool output → shown as "Tool:"
   AND appended to `_chat_history` with role "tool"). Storing tool output
   in history is the key organizer step: your next message ("what could
   cause these?") reaches the model with the exact scan numbers in
   context, enabling cause analysis without recomputation. Empty replies
   that contain only actions display as "(actions only)".

3. **New "Detect anomalies" button** in the AI button row. Why: the scan
   is deterministic, so you shouldn't need an LLM round-trip (or even a
   model installed) to run it. It posts the report to the chat and
   history, then the status bar suggests asking the chat for causes.

4. **`run_anomaly_scan()` method** implementing that button via the same
   `run_tool()` path the model uses — one code path, two entry points,
   so button results and model-triggered results are always identical.

5. **`_set_ai_busy()` includes the new button** so it greys out during
   model calls like its siblings.

============================================================
## Change Set 06 — requirements.txt
- Added `scipy>=1.11`. Why: signal_tools' Butterworth `lowpass()` and
  any future filtering in formulas silently fall back to the slower RC
  approximation without it; the detector itself stays NumPy-only.

============================================================
## Verification performed (headless, offscreen Qt)
- T0001.CSV loads; scan reports: CH1 → "301 periodic transition events,
  median spacing 0.5001 ms (~2 per ms)… PWM edges" (correct: 1 kHz
  control, 2 edges/period); CH2 → 37 spike events ~769 kHz ringing +
  crest factor 6.1 flag; synthetic random spikes are NOT classified as
  periodic.
- Model-reply simulation with mixed formatting + tool actions: title and
  x-range applied, tool ran, history roles = [tool, assistant, tool,
  tool], plot refreshed.

## Suggested next steps (not implemented)
- Auto-interpret toggle: after a button scan, optionally auto-send one
  model turn ("explain likely causes") — trivial with the existing
  pieces, left off to keep model calls explicit.
- Imbalance detector currently keys on "(A)"/"current" in labels; a
  per-channel "is module current" checkbox would make S1…S4 grouping
  explicit.
- Persist anomaly reports alongside exports for your Obsidian workflow.
