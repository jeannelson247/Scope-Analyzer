#!/usr/bin/env bash
# Stop the two local MLX-LM endpoints started by start_dual_mlx_agents.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${SCOPE_STUDIO_AGENT_RUNTIME:-"$ROOT/.agent_runtime"}"

stop_one() {
  local role="$1"
  local pidfile="$RUNTIME_DIR/${role}.pid"
  if [[ ! -f "$pidfile" ]]; then
    printf '%s: no pidfile\n' "$role"
    return 0
  fi
  local pid
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" >/dev/null 2>&1; then
    printf 'Stopping %s pid=%s\n' "$role" "$pid"
    kill "$pid" || true
  else
    printf '%s pid=%s is not running\n' "$role" "$pid"
  fi
  rm -f "$pidfile"
}

stop_one planner
stop_one coder
