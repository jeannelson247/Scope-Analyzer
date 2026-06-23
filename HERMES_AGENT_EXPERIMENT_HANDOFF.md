# Scope Studio Hermes Agent Experiment Handoff

Date: 2026-06-23

Purpose: test whether a local Hermes + MLX agent can help develop Scope Studio without
breaking the scientific core, raw-data immutability, or the user-facing Lite workflow.

## Current Project State

- Canonical project: `/Users/jeannelson/Desktop/scope_studio03`
- Safe experiment copy: `/Users/jeannelson/Desktop/scope_studio03_hermes_lab`
- External model vault: `/Volumes/JeanDrive1/Models`
- MLX model folder: `/Volumes/JeanDrive1/Models/mlx`
- Primary native app: `app.py`
- Lite web app: `scope_web/index.html` + `scope_web/backend_api.py`
- Deterministic scientific core: `csv_loader.py`, `signal_tools.py`,
  `calibration.py`, `saturation_recovery.py`, `rlc_reconstruct.py`,
  `reconstruction_audit.py`, `data_quality.py`, `detect_anomalies.py`

## Non-Negotiable Guardrails

- Never overwrite or modify original CSV/TXT source files.
- Never rewrite the scientific core without an explicit human-approved task.
- All formulas, filters, reconstructions, overlays, and generated estimates are
  in-memory display transforms unless the user explicitly exports a new file.
- Derived CSVs, figures, logs, or reports must be new files.
- No autonomous push to GitHub. The agent may create a local branch, patch, or
  PR draft, but the user reviews before any remote update.
- If a task touches RLC reconstruction, calibration, unit conversions, source CSV
  parsing, or validation benchmarks, run the relevant tests before reporting done.

## Recommended Two-Model Collaboration Protocol

Use two models as a relay, not as competitors. This is now implemented as a
RecursiveMAS-inspired text/JSON packet protocol in:

- `recursive_agent_protocol.py`
- `scripts/recursive_agent_packet.py`
- `docs/RECURSIVE_MAS_SCOPE_STUDIO.md`

Generate a compact prompt for each local model instead of pasting the whole chat:

```bash
python scripts/recursive_agent_packet.py \
  --task "Audit Lite web app coverage against app.py without editing code." \
  --role planner \
  --phase plan \
  --out-dir agent_packets
```

1. Planner/reviewer model reads the task, inspects files, and writes a short
   implementation plan with risk notes and tests.
2. Coder model implements only the approved plan in small commits or patches.
3. Planner/reviewer model reviews the diff for logic, safety, and UI consistency.
4. Tests and benchmarks decide pass/fail. If tests fail, the coder gets the
   exact failure and fixes only that failure.
5. A final handoff note records changed files, why they changed, commands run,
   and remaining risks.

Avoid letting both models edit the same files in parallel. The clean pattern is:
planner -> coder -> tests -> reviewer -> user.

Why this is a RecursiveMAS adaptation:
- Full RecursiveMAS passes latent states through trained RecursiveLink adapters.
- Scope Studio currently uses Hermes + MLX-LM, so we use compact packets as a
  practical latent-state proxy: decisions, changed files, risks, tests, and next
  action only.
- This keeps communication cheap and prevents two models from re-reading or
  re-arguing the same long transcript.

## Recommended Model Roles On This Mac

Machine context: MacBook Pro M4 Pro, 24 GB RAM. Keep routine models under roughly
18 GB so the app, browser, test runner, and MLX server can coexist.

- Default coder: `/Volumes/JeanDrive1/Models/mlx/Qwen2.5-Coder-14B-Instruct-4bit`
  - Best balance for code integration, refactors, tests, and file organization.
- Fast planner/router: `/Volumes/JeanDrive1/Models/mlx/Qwen3.5-9B-MLX-4bit`
  - Use for issue triage, test planning, UI/text cleanup, and small patches.
- Heavy experimental reviewer:
  `/Volumes/JeanDrive1/Models/mlx/Qwen3-Coder-30B-A3B-Instruct-4bit`
  - Use sparingly for architecture review or hard refactor proposals; it is large
    enough to be slower/tighter on 24 GB.
- Lightweight fallback:
  `/Volumes/JeanDrive1/Models/mlx/Qwen2.5-Coder-3B-Instruct-4bit`
  - Use for very fast routing or student-laptop defaults.

The LLM should not do final numerical interpretation from raw samples. It should
call deterministic tools, explain their results, and propose code changes.

## MLX Server For Hermes

For one active model, start one local MLX server:

```bash
cd /Users/jeannelson/Desktop/scope_studio03_hermes_lab
source venv/bin/activate
python -m mlx_lm.server \
  --model /Volumes/JeanDrive1/Models/mlx/Qwen2.5-Coder-14B-Instruct-4bit \
  --port 8080
```

Then configure Hermes to use the local OpenAI-compatible endpoint:

```bash
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8080/v1
hermes config set model.default Qwen2.5-Coder-14B-Instruct-4bit
```

If Hermes requires the exact model identifier, use the full local model path.

For two simultaneous models, use two MLX-LM endpoints:

```bash
cd /Users/jeannelson/Desktop/scope_studio03_hermes_lab
source venv/bin/activate
scripts/start_dual_mlx_agents.sh
```

This starts:
- planner/reviewer: `http://127.0.0.1:8081/v1`
- coder: `http://127.0.0.1:8082/v1`

Stop them with:

```bash
scripts/stop_dual_mlx_agents.sh
```

Read `docs/DUAL_MLX_HERMES_SETUP.md` for the manual-switching, two-session, and
coordinator options. The safest rule remains: only one agent edits at a time.

For the lower-memory Orchestra-style workflow, prefer sequential model swapping:

```bash
/Users/jeannelson/Desktop/scope_studio03/venv/bin/python scripts/scope_agent_turn.py \
  --role planner \
  --task "Audit Lite app coverage against app.py without editing code." \
  --phase plan
```

This starts one model, sends one compact packet, saves the specialist response,
then unloads the model and returns control to the main chat/overseer. See
`docs/SEQUENTIAL_MODEL_SWAP_COORDINATOR.md`.

Note: the Hermes lab copy may not have its own `venv` because copied virtual
environments are large and brittle. Using the canonical venv path above runs the
lab code while reusing the already-installed MLX/PySide/SciPy environment.

## Suggested First Agent Task

Give Hermes this bounded task:

```text
You are working in /Users/jeannelson/Desktop/scope_studio03_hermes_lab.
Do not touch /Users/jeannelson/Desktop/scope_studio03.
Do not push to GitHub.
Do not rewrite scientific core modules.

Task:
Audit the Lite web app and identify which original Scope Studio functions from
app.py are not yet exposed through scope_web/index.html and scope_web/backend_api.py.
Produce a markdown checklist grouped by:
1. already implemented,
2. missing but safe to port,
3. missing and requires human review because it affects scientific assumptions.

Do not edit code on this pass. Run only read-only inspection commands.
```

## Required Verification Commands

Run these after code edits:

```bash
python -m py_compile ai_assistant.py scope_web/backend_api.py app.py
python -m pytest -q
python scripts/benchmark_lite_toolbox.py
python scripts/benchmark_lite_stress_tools.py
```

Last coordinator verification on 2026-06-23:

```bash
python3 -m py_compile scope_agent_turn.py scripts/scope_agent_turn.py \
  recursive_agent_protocol.py scripts/recursive_agent_packet.py
python3 -m pytest tests/test_scope_agent_turn.py \
  tests/test_recursive_agent_protocol.py -q
python3 -m pytest -q
```

Result: coordinator focused tests passed `7 passed`; full suite passed
`96 passed, 1 skipped`.

Follow-up verification after the local-model payload fix:

```bash
venv/bin/python scripts/scope_agent_turn.py \
  --role summarizer \
  --model /Volumes/JeanDrive1/Models/mlx/Qwen2.5-Coder-3B-Instruct-4bit \
  --task "Smoke test only. Reply with one sentence: MLX sequential specialist handoff works." \
  --phase handoff \
  --max-tokens 64
python3 -m pytest -q
```

Result: the 3B MLX specialist loaded, responded, wrote turn artifacts, and
unloaded successfully (`ok=true`, `stopped_model=true`, about 5.2 s). Full suite
then passed `97 passed, 1 skipped`.

For Lite packaging:

```bash
./scope_analyzer/lite/build.sh
codesign --verify --deep --strict dist/ScopeAnalyzerLite.app
```

## Notes For Future AI Collaborators

- Read `DEVELOPMENT_LOG_SCOPE_STUDIO.txt` before editing. It is the chronological
  lab notebook.
- Read `SCOPE_STUDIO_HANDOFF.md` for architecture and the web-vs-native decision.
- The current app intentionally develops native Qt and Lite web in tandem. Do not
  merge them into one giant file.
- If a feature can be deterministic, put it behind `scope_web/backend_api.py` so
  Lite users without an LLM can still use it.
- Keep the interface beginner-friendly: tools should have small parameter forms,
  defaults, and explanatory reports.
