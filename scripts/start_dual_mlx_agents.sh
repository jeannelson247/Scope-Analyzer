#!/usr/bin/env bash
# Start two local MLX-LM OpenAI-compatible endpoints for Scope Studio agents.
#
# Planner endpoint: compact planning/review packets.
# Coder endpoint: implementation packets.
#
# This does not grant either model filesystem access by itself. Hermes or another
# tool-runner must still decide what actions are allowed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${SCOPE_STUDIO_AGENT_RUNTIME:-"$ROOT/.agent_runtime"}"
mkdir -p "$RUNTIME_DIR"

PYTHON_BIN="${PYTHON_BIN:-"$ROOT/venv/bin/python"}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

PLANNER_MODEL="${PLANNER_MODEL:-/Volumes/JeanDrive1/Models/mlx/Qwen3.5-9B-MLX-4bit}"
CODER_MODEL="${CODER_MODEL:-/Volumes/JeanDrive1/Models/mlx/Qwen2.5-Coder-14B-Instruct-4bit}"
PLANNER_PORT="${PLANNER_PORT:-8081}"
CODER_PORT="${CODER_PORT:-8082}"
HOST="${MLX_AGENT_HOST:-127.0.0.1}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

check_model() {
  local role="$1"
  local path="$2"
  [[ -d "$path" ]] || die "$role model folder not found: $path"
  [[ -f "$path/config.json" ]] || die "$role model missing config.json: $path"
}

port_is_listening() {
  local port="$1"
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

start_server() {
  local role="$1"
  local model="$2"
  local port="$3"
  local log="$RUNTIME_DIR/${role}.log"
  local pidfile="$RUNTIME_DIR/${role}.pid"

  if port_is_listening "$port"; then
    printf '%s endpoint already listening at http://%s:%s/v1\n' "$role" "$HOST" "$port"
    return 0
  fi

  printf 'Starting %s endpoint on http://%s:%s/v1\n' "$role" "$HOST" "$port"
  printf '  model: %s\n' "$model"
  nohup "$PYTHON_BIN" -m mlx_lm server \
    --host "$HOST" \
    --port "$port" \
    --model "$model" \
    --max-tokens 2048 \
    --temp 0.0 \
    >"$log" 2>&1 &
  echo $! >"$pidfile"
  sleep 2

  if port_is_listening "$port"; then
    printf '%s ready. pid=%s log=%s\n' "$role" "$(cat "$pidfile")" "$log"
  else
    printf '%s did not start yet. Check log: %s\n' "$role" "$log" >&2
    tail -n 30 "$log" >&2 || true
    return 1
  fi
}

check_model "planner" "$PLANNER_MODEL"
check_model "coder" "$CODER_MODEL"

start_server "planner" "$PLANNER_MODEL" "$PLANNER_PORT"
start_server "coder" "$CODER_MODEL" "$CODER_PORT"

cat > "$RUNTIME_DIR/endpoints.env" <<EOF
PLANNER_ENDPOINT=http://$HOST:$PLANNER_PORT/v1
PLANNER_MODEL=$PLANNER_MODEL
CODER_ENDPOINT=http://$HOST:$CODER_PORT/v1
CODER_MODEL=$CODER_MODEL
EOF

cat <<EOF

Dual MLX endpoints are configured:
  planner: http://$HOST:$PLANNER_PORT/v1
  coder:   http://$HOST:$CODER_PORT/v1

Endpoint metadata written to:
  $RUNTIME_DIR/endpoints.env

Stop them with:
  scripts/stop_dual_mlx_agents.sh
EOF
