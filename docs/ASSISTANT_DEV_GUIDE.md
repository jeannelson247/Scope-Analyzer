# Building the Bespoke Lab Assistant — Development Guideline

*How Jean's personal assistant gets sharper over time without ever
becoming less trustworthy. Companion to PROJECT_REPORT.md.*

## The target profile

A sharp problem solver, brilliant pattern identifier, and logical
deduction machine — that **routes** to deterministic tools for every
number, **drafts** new tools when a gap appears, and **learns** the
user's rig, vocabulary, and preferences through curated context.

## Seats, not one model

| Seat | Job | Selection gate (benchmark family) | Current pick |
|---|---|---|---|
| Router | message → tool JSON, ~0.2 s | A/B + E (routing) | qwen2.5:0.5b (Ollama) / Llama-3.2-3B (MLX) |
| Interpreter | explain tool output, discuss | reasoning bench (P/N/DH/DA) | Qwen3.5-9B-MLX |
| Coder | draft tools | D/H + O (creation, optimization) | Qwen2.5-Coder-7B (30B MoE for heavy sessions) |
| Analyst | review/optimize drafts, math | personal-assistant bench (L/R/S/O/A) | *open — decided by this benchmark* |

Rules learned the hard way (all measured, all in the dev log):
math-derivation models never sit in the interpreter or router seat;
R1-style reasoners never route; every seat assignment cites a benchmark
table, not vibes.

## The selection loop (run after any model download)

1. `benchmark_mlx_models.py` — role triage over the whole vault.
2. `benchmark_tool_creation.py` — D/E/F/G/H for coder + selector seats.
3. `benchmark_reasoning.py` — fair thinking-budget physics/numerics.
4. `benchmark_personal_assistant.py` — L/R/S/O/A novel-environment
   reasoning + measured optimization + the ungraded open-conjecture
   probe (read the probe transcripts yourself; structure and honesty
   about uncertainty matter more than the answer).
5. Update `model_catalog.py` and USER_DEFAULTS; log the change set.

## The improvement ladder (cheapest first)

1. **Context**: curate lab_memory ("remember:" + "compress memory") —
   instant, reviewable, reversible.
2. **Prompts**: few-shot examples in router/tool prompts; add aliases
   when a routing miss is observed.
3. **Tools**: gaps become tool_sandbox drafts via the orchestrator
   (coder drafts → analyst revises → tests pick → human promotes →
   dev-log entry).
4. **Fine-tuning (last)**: when a *format/behavior* (not facts) refuses
   to stick, a LoRA via mlx-lm on the lab_memory journal + chat history
   as the dataset. Never tune facts — facts live in tools and memory.

## Benchmark-design principles (for adding tasks)

- Novel framing or invented notation — memorization must not pay.
- Every graded answer machine-checkable (number, letter, exact
  Fraction, or measured speedup); generated code only ever runs
  screened + sandboxed.
- Compute truths in the script where possible (the iterated-map task
  simulates its own answer).
- Reasoning models get thinking budgets; time is reported, not scored.
- Keep one UNGRADED probe per suite: open problems calibrate *judgment*
  — a model that confidently "solves" an open conjecture is telling
  you something important about its honesty.
- Validate every suite with mock perfect/naive before trusting it on
  real models.

## Cadence

Per session: run the shot pipeline, note misses, "remember:" the fixes.
Per new model: the selection loop. Per release: full regression
(backtests + all mock suites) before tagging.
