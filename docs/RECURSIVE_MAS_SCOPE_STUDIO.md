# RecursiveMAS-Inspired Scope Studio Agent Workflow

Date: 2026-06-23

Source context:
- Project page: https://recursivemas.github.io/
- GitHub: https://github.com/RecursiveMAS/RecursiveMAS
- Paper: https://arxiv.org/abs/2604.25917

## What We Are Borrowing

RecursiveMAS shows that multi-agent systems can become more efficient when agents
do not repeatedly exchange long text transcripts. The paper's full method uses
trained RecursiveLink modules to pass latent states between agents. That is not
the right first implementation for Scope Studio because our active stack is
Hermes Agent + MLX-LM + deterministic Python tools, not a PyTorch training
pipeline with released RecursiveLink checkpoints converted to MLX.

Instead, Scope Studio borrows the system pattern:

- Sequential style: planner -> coder -> reviewer.
- Mixture style: optional specialist opinions for UI, science, and tests.
- Distillation style: large model drafts guidance; small model executes routine
  steps when possible.
- Deliberation style: reflector verifies logic; tool-caller runs deterministic
  commands.

The efficient substitute for latent transfer is a compact JSON/Markdown packet
that preserves only decisions, changed files, risks, tests, and next action.

## Implemented Local Protocol

Files:
- `recursive_agent_protocol.py`
- `scripts/recursive_agent_packet.py`
- `tests/test_recursive_agent_protocol.py`

The protocol generates role-specific packets for local agents:

```bash
python scripts/recursive_agent_packet.py \
  --task "Audit Lite web app coverage against app.py without editing code." \
  --role planner \
  --phase plan \
  --out-dir agent_packets
```

Then pass the generated `.md` prompt to Hermes or another local model.

## Recommended Roles On Jean's Mac

- Planner/reviewer: `/Volumes/JeanDrive1/Models/mlx/Qwen3.5-9B-MLX-4bit`
- Main coder: `/Volumes/JeanDrive1/Models/mlx/Qwen2.5-Coder-14B-Instruct-4bit`
- Heavy reviewer only when needed:
  `/Volumes/JeanDrive1/Models/mlx/Qwen3-Coder-30B-A3B-Instruct-4bit`
- Lightweight fallback: `/Volumes/JeanDrive1/Models/mlx/Qwen2.5-Coder-3B-Instruct-4bit`

Run one model at a time on 24 GB RAM unless you have confirmed both can stay
resident without memory pressure. Communication efficiency comes from concise
handoffs, not from keeping every model loaded.

## Why This Should Work

For Scope Studio, most failures are not caused by lack of model IQ. They come
from context drift, unclear ownership, hidden file edits, and insufficient tests.
The packet protocol attacks exactly those issues:

- It limits the model's prompt to the current task state.
- It tells each model its role and forbids role drift.
- It lists protected scientific files every time.
- It makes tests the arbitration layer.
- It compresses each recursion round into reusable notes instead of preserving a
  giant chat transcript.

## What We Are Not Doing Yet

- No trained RecursiveLink modules.
- No hidden-state exchange between MLX models.
- No automatic GitHub push.
- No local agent rewrites of `rlc_reconstruct.py`, `calibration.py`, or CSV
  loading without explicit human approval.

That can be revisited later if MLX-compatible RecursiveMAS checkpoints or a
practical adapter-training path becomes available.
