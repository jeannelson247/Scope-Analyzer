# Sequential Model-Swap Coordinator

Date: 2026-06-23

This is the safest architecture for Jean's 24 GB M4 Pro:

```text
persistent chat / overseer
    -> create compact packet
    -> load one specialist model
    -> specialist responds
    -> unload specialist
    -> overseer reviews response and decides the next step
```

The persistent chat AI is the coordinator. The local MLX models are specialists
loaded only when needed. This mirrors the Orchestra rule: model proposes,
verifier decides, human approves irreversible changes.

## Why Prefer This Over Two Always-On Models

- Lower memory pressure.
- Less chance of macOS swapping.
- Easier to keep roles clean.
- The coordinator remains the single source of objective/state.
- Each specialist turn leaves a prompt, packet, response, and metadata trail.

## Run One Planner Turn

```bash
cd /Users/jeannelson/Desktop/scope_studio03_hermes_lab
python scripts/scope_agent_turn.py \
  --role planner \
  --task "Audit Lite app coverage against app.py without editing code." \
  --phase plan
```

If the Hermes lab copy does not have its own `venv`, run the same lab code with
the canonical Scope Studio environment:

```bash
cd /Users/jeannelson/Desktop/scope_studio03_hermes_lab
/Users/jeannelson/Desktop/scope_studio03/venv/bin/python scripts/scope_agent_turn.py \
  --role planner \
  --task "Audit Lite app coverage against app.py without editing code." \
  --phase plan
```

The runner will:

1. Build a compact RecursiveMAS-style packet.
2. Start one MLX-LM server on `http://127.0.0.1:8091/v1`.
3. Send the packet to the role model.
4. Save artifacts under `.agent_runtime/turns/<timestamp>_planner/`.
5. Stop the MLX server.

## Run One Coder Turn

Use this only after the coordinator approves a plan:

```bash
python scripts/scope_agent_turn.py \
  --role coder \
  --task "Implement only the approved Lite UI checklist item. Do not touch scientific core." \
  --phase implement
```

The coder endpoint still only generates text. It does not edit files by itself.
Hermes/Codex/the user decides whether to apply a patch.

## Run A Reviewer Turn

After edits and tests:

```bash
python scripts/scope_agent_turn.py \
  --role reviewer \
  --task "Review the current diff for safety, missing tests, UI regressions, and protected-core violations." \
  --phase review
```

## How This Works With Hermes

Hermes can remain the action-taking environment, while the local model turn
runner is treated like a specialist-call tool:

1. Hermes/main chat decides the role and task.
2. It runs `scripts/scope_agent_turn.py`.
3. It reads the generated response.
4. It verifies logic and tests.
5. It either asks another specialist, applies a small patch, or stops for Jean.

## Important Limits

- Do not let both Hermes and the specialist runner edit files at the same time.
- Do not run this with a huge model while the GUI, tests, and another model are
  already resident.
- Do not accept a specialist response as truth. It is a proposal.

## Artifact Layout

Each turn writes:

```text
.agent_runtime/turns/YYYYMMDD_HHMMSS_role/
  packet.json
  prompt.md
  response.md
  metadata.json
```

These artifacts become the recursion memory for the next round.
