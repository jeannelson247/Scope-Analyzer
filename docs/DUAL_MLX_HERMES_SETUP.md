# Dual MLX Models With Hermes

Date: 2026-06-23

Goal: run two local models for Scope Studio agent experiments without making the
models compete or edit the same files.

## Short Answer

Yes, two models can run simultaneously, but not as one magic shared brain. The
clean setup is:

- one MLX-LM server for the planner/reviewer model,
- one MLX-LM server for the coder model,
- Hermes or a small coordinator routes compact RecursiveMAS-style packets between
  them.

On a 24 GB M4 Pro, this is useful for experiments but should not be the default
daily workflow. If memory pressure appears, run one model at a time and keep the
same packet protocol.

## Start Two MLX Endpoints

From the project or Hermes lab copy:

```bash
cd /Users/jeannelson/Desktop/scope_studio03_hermes_lab
source venv/bin/activate
scripts/start_dual_mlx_agents.sh
```

Defaults:

- Planner/reviewer model:
  `/Volumes/JeanDrive1/Models/mlx/Qwen3.5-9B-MLX-4bit`
- Coder model:
  `/Volumes/JeanDrive1/Models/mlx/Qwen2.5-Coder-14B-Instruct-4bit`
- Planner endpoint: `http://127.0.0.1:8081/v1`
- Coder endpoint: `http://127.0.0.1:8082/v1`

Stop them:

```bash
scripts/stop_dual_mlx_agents.sh
```

Override defaults:

```bash
PLANNER_MODEL=/Volumes/JeanDrive1/Models/mlx/Qwen2.5-Coder-3B-Instruct-4bit \
CODER_MODEL=/Volumes/JeanDrive1/Models/mlx/Qwen2.5-Coder-14B-Instruct-4bit \
PLANNER_PORT=8081 \
CODER_PORT=8082 \
scripts/start_dual_mlx_agents.sh
```

## How Hermes Fits

Hermes can point to an OpenAI-compatible endpoint. For one active model:

```bash
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8082/v1
hermes config set model.default Qwen2.5-Coder-14B-Instruct-4bit
```

For two models, use one of these patterns:

1. Manual switching:
   - use planner endpoint for planning/review,
   - switch to coder endpoint for implementation.
2. Two Hermes sessions/profiles, if your Hermes install supports profile/config
   isolation:
   - planner session points at `8081`,
   - coder session points at `8082`.
3. Coordinator pattern:
   - generate compact packets using `scripts/recursive_agent_packet.py`,
   - send planner packets to `8081`,
   - send coder packets to `8082`,
   - let tests decide whether the round passes.

The coordinator pattern is the closest practical analogue to RecursiveMAS for
our MLX setup.

## Recommended Communication Contract

Do not let both agents edit files at the same time.

Use:

```text
planner -> coder -> tests -> reviewer -> user
```

The planner/reviewer verifies logic before implementation. The coder implements
only the approved plan. Tests arbitrate. The user approves before GitHub.

## Generate A Planner Packet

```bash
python scripts/recursive_agent_packet.py \
  --task "Audit Lite app coverage against app.py without editing code." \
  --role planner \
  --phase plan \
  --out-dir agent_packets
```

Give the generated `.md` file to the planner model.

## Memory Notes

Two simultaneous models can be heavy:

- 9B + 14B should usually be plausible on 24 GB, but leave headroom for the app,
  browser, tests, and macOS.
- 30B-A3B plus another model may be too tight for stable daily work.
- If macOS starts swapping, stop one endpoint and keep the relay protocol.

## Safety Notes

- The MLX servers only generate text. They do not edit files by themselves.
- Hermes/tool access is the dangerous part, so keep the protected-file guardrails
  in every packet.
- Never give both sessions unrestricted write permission to the same working tree.
