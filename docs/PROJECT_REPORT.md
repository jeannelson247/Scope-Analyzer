# Scope Studio — Project Report

*Prepared for the v0.1 open-source release. Companion documents:
[USER_GUIDE.md](USER_GUIDE.md) (the instruction manual),
[SECURITY.md](../SECURITY.md) (privacy & trust),
DEVELOPMENT_LOG_SCOPE_STUDIO.txt (the complete engineering record,
50 logged change sets).*

## The goal

Scope Studio began as one scientist's need: analyze oscilloscope shots
from tokamak TF-coil current-driver tests — quickly, correctly, and
without writing code at the test bench. It grew into a general aim:

> A local-first data-analysis workbench where deterministic, tested
> mathematics does the computing, a small local AI does the routing and
> explaining, every number is traceable to its source file, and a
> student with no coding background gets value in their first minute.

Two tracks ship from one codebase: the **full** version (tuned for an
Apple Silicon workstation with local 4–30B models) and a **lite**
version (≤2 GB models, no MoEs) for students with limited hardware.

## The core design rule

**The AI never computes numbers.** This is not a slogan; it is a
measured result. In our benchmark, every local model from 0.5B to 30B
scored 0/3 computing averages from 48 raw values — several could not
find the maximum — while small models scored perfectly at *reading*
computed statistics and *routing* to deterministic tools. NumPy/SciPy
compute; the model reads, routes, and explains. Every analytical claim
in the app traces to deterministic code with a test against known truth.

## What was built (by theme)

**Data layer.** Robust instrument-CSV loader (Tektronix/Rigol/LeCroy
preambles, any delimiter, 1.25M rows in <1 s, float32);
SHA-256-verified read-only sessions; derived results saved only as
explicit `_analyzed.csv` copies; min–max decimation that preserves
spikes.

**Deterministic toolset** (each verified against synthetic ground
truth; errors quoted from the tests): channel statistics; anomaly
scanning (MAD spikes with ringing-frequency estimate, clipping, drift,
module imbalance); baseline zeroing; saturation estimation (two-slope
censored fit; soft-saturation mode with guard band — 0.0% peak error on
synthetic truth); censored-ML RLC reconstruction (multi-sensor,
Tobit-style lower bounds, bootstrap bands — 0.3% max error across a
fully censored 35 ms gap); post-edge derivative-spectrum ringing
detection (138 kHz burst on a 5 kA edge detected at 134× the false-
positive floor); calibration cross-fits.

**Visualization.** pyqtgraph 2D with dual axes, Wong palette,
trigger-aligned multi-shot overlays, fit overlays with legends; tabbed
3D window (matplotlib surfaces with auto-binning, shot waterfalls, V–I
trajectory maps, full-resolution Detail+FFT view); GPU renderer
(pyqtgraph.opengl) for million-point cascades; Nature-style SVG/JPG
export; clipboard copy via double-click (PNG/JPG/SVG).

**Local AI.** Three backends (MLX direct, Ollama, llama.cpp); two-tier
chat — a small router (~0.2 s, schema-constrained where supported)
dispatches tool requests, a larger interpreter handles discussion;
display-state undo for every AI action; lab-memory journal + compressed
digest carried in every prompt; Obsidian session notes with a
deterministic "tools used, and why" report; an inactive-draft tool
sandbox plus a two-model orchestrator (coder drafts, analyst revises,
tests pick the winner, human promotes).

**Verification culture.** Five benchmark suites (stats-reading/routing/
arithmetic; tool-creation/algorithm-selection/guided-physics/reasoning/
algorithms; a fair reasoning benchmark with thinking budgets; a
role-aware MLX fleet benchmark; real-data backtests against the 6.6 kA
campaign). The development log records every change with what/why/how-
verified — including the failures (four iterations to a trustworthy
ringing detector are documented, deliberately).

## Verification highlights

| Claim | Evidence |
|---|---|
| Loader + calibration correct on real data | 6.6 kA nameplate vs measured Pearson peak: 0.3%; charge constant (21.8–25.8 C) across the 14-shot step series |
| Saturation recovery works | 0.06% peak error (hard clip), 0.0–1% (soft clip), negative controls pass |
| RLC reconstruction works | 0.3% max error over the censored 5–40 ms gap, consistent with both sensors; on the real shot: τ_rise 2.29 ms vs L/R = 2 ms predicted |
| Small models suffice for routing | 0.5B model: 4/4 stats-reading, 2/2 routing; arithmetic 0/3 at every size |
| Sandboxed code-gen is gated | banned-construct screen catches injected `import os`; isolated subprocess execution |

## Known limitations

Single-shot RLC model assumes overdamped free discharge (fit windows
must exclude switch-off); Pearson CTs saturate on long pulses (use the
early window); model quality varies by seat — *math-derivation models
must not be used as chat interpreters* (they answer with derivations,
not tool calls; see USER_GUIDE model table); executables must be built
per-OS; live PXI streaming and the web frontend are roadmap items.

## Roadmap

Module split (`core`/`ui`/`tools`) → pytest harvest + CI matrix →
batch shot journal + PPTX reports → packaged installers (macOS .app,
Windows .exe for PXI) → scope-screenshot digitizer (last-resort data
recovery) → web frontend over the same core.
