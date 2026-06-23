"""Load one local MLX specialist for one bounded Scope Studio agent turn.

The persistent chat/overseer stays outside this script. This runner starts one
MLX-LM endpoint, sends a compact RecursiveMAS-style packet, saves the response,
and stops the endpoint unless told otherwise. It is designed for 24 GB Macs:
one specialist resident at a time, with explicit artifacts for review.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from recursive_agent_protocol import Role, build_packet, role_prompt


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNTIME = ROOT / ".agent_runtime"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8091
DEFAULT_TIMEOUT_S = 420.0

DEFAULT_MODELS = {
    "planner": "/Volumes/JeanDrive1/Models/mlx/Qwen3.5-9B-MLX-4bit",
    "reviewer": "/Volumes/JeanDrive1/Models/mlx/Qwen3.5-9B-MLX-4bit",
    "coder": "/Volumes/JeanDrive1/Models/mlx/Qwen2.5-Coder-14B-Instruct-4bit",
    "summarizer": "/Volumes/JeanDrive1/Models/mlx/Qwen2.5-Coder-3B-Instruct-4bit",
}

SYSTEM_PROMPTS = {
    "planner": (
        "You are the Scope Studio planning specialist. Inspect the packet, "
        "produce a concise implementation plan, list risks, and specify tests. "
        "Do not claim to have edited files."
    ),
    "coder": (
        "You are the Scope Studio coding specialist. Produce a minimal patch plan "
        "or code-oriented instructions for the coordinator. Do not invent test "
        "results. Respect protected scientific modules."
    ),
    "reviewer": (
        "You are the Scope Studio reviewer. Look for bugs, unsafe edits, missing "
        "tests, and violations of guardrails. Findings first."
    ),
    "summarizer": (
        "You are the Scope Studio handoff compressor. Preserve only decisions, "
        "changed files, test results, blockers, and the next action."
    ),
}


@dataclass
class TurnResult:
    ok: bool
    role: str
    model: str
    endpoint: str
    out_dir: str
    response_path: str = ""
    prompt_path: str = ""
    packet_path: str = ""
    error: str = ""
    elapsed_s: float = 0.0
    stopped_model: bool = False


def default_model_for_role(role: str) -> str:
    return DEFAULT_MODELS.get(role, DEFAULT_MODELS["planner"])


def endpoint_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


def model_payload_id(model: str) -> str:
    """Use the local folder path for MLX servers backed by local model folders."""
    model_path = Path(model).expanduser()
    return str(model_path) if model_path.exists() else model


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _server_alive(endpoint: str) -> bool:
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=2) as resp:
            return resp.status < 500
    except Exception:
        return False


def wait_for_server(endpoint: str, timeout_s: float = 120.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if _server_alive(endpoint):
            return
        time.sleep(1.0)
    raise TimeoutError(f"MLX endpoint did not become ready: {endpoint}")


def start_mlx_server(
    model: str,
    *,
    host: str,
    port: int,
    runtime_dir: Path,
    python_bin: str,
    max_tokens: int,
) -> subprocess.Popen:
    model_path = Path(model).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model folder not found: {model}")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "active_mlx_server.log"
    log = log_path.open("a", encoding="utf-8")
    cmd = [
        python_bin,
        "-m",
        "mlx_lm",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        str(model_path),
        "--max-tokens",
        str(max_tokens),
        "--temp",
        "0.0",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def stop_process(proc: subprocess.Popen | None) -> bool:
    if proc is None or proc.poll() is not None:
        return False
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=10)
    return True


def chat_completion(endpoint: str, *, model: str, system_prompt: str, prompt: str, timeout_s: float) -> str:
    payload = {
        "model": model_payload_id(model),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    data = _post_json(endpoint.rstrip("/") + "/chat/completions", payload, timeout_s)
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"unexpected chat completion response: {data!r}") from exc


def build_prompt_from_args(args: argparse.Namespace) -> tuple[str, str]:
    if args.prompt_file:
        prompt_path = Path(args.prompt_file).expanduser()
        return prompt_path.read_text(encoding="utf-8"), "{}"
    if args.packet_json:
        packet_data = _read_json(Path(args.packet_json).expanduser())
        # Re-render the prompt with current role rule, but preserve packet data.
        from recursive_agent_protocol import AgentPacket

        packet = AgentPacket(**packet_data)
        return role_prompt(packet, args.role), packet.to_json()
    packet = build_packet(
        args.project_root,
        args.task,
        phase=args.phase,
        round_index=args.round,
        max_rounds=args.max_rounds,
        state_summary=args.state,
        decisions=args.decision,
        open_questions=args.question,
        next_action=args.next_action,
    )
    return role_prompt(packet, args.role), packet.to_json()


def run_turn(args: argparse.Namespace) -> TurnResult:
    t0 = time.time()
    role = args.role
    model = args.model or default_model_for_role(role)
    host = args.host
    port = args.port
    endpoint = args.endpoint or endpoint_url(host, port)
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else runtime_dir / "turns" / f"{stamp}_{role}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt, packet_json = build_prompt_from_args(args)
    prompt_path = out_dir / "prompt.md"
    packet_path = out_dir / "packet.json"
    response_path = out_dir / "response.md"
    metadata_path = out_dir / "metadata.json"
    _write(prompt_path, prompt)
    _write(packet_path, packet_json)

    proc: subprocess.Popen | None = None
    stopped = False
    try:
        if not args.no_start_server:
            if _server_alive(endpoint):
                raise RuntimeError(
                    f"endpoint already alive at {endpoint}; pass --no-start-server "
                    "to reuse it or choose a different --port"
                )
            proc = start_mlx_server(
                model,
                host=host,
                port=port,
                runtime_dir=runtime_dir,
                python_bin=args.python_bin,
                max_tokens=args.max_tokens,
            )
            wait_for_server(endpoint, timeout_s=args.start_timeout)
        text = chat_completion(
            endpoint,
            model=model,
            system_prompt=SYSTEM_PROMPTS[role],
            prompt=prompt,
            timeout_s=args.timeout,
        )
        _write(response_path, text)
        result = TurnResult(
            ok=True,
            role=role,
            model=model,
            endpoint=endpoint,
            out_dir=str(out_dir),
            response_path=str(response_path),
            prompt_path=str(prompt_path),
            packet_path=str(packet_path),
            elapsed_s=time.time() - t0,
        )
    except Exception as exc:
        result = TurnResult(
            ok=False,
            role=role,
            model=model,
            endpoint=endpoint,
            out_dir=str(out_dir),
            prompt_path=str(prompt_path),
            packet_path=str(packet_path),
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.time() - t0,
        )
    finally:
        if not args.keep_server:
            stopped = stop_process(proc)
        result.stopped_model = stopped
        metadata = asdict(result)
        _write(metadata_path, json.dumps(metadata, indent=2, sort_keys=True))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["planner", "coder", "reviewer", "summarizer"], required=True)
    parser.add_argument("--task", default="")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--phase", choices=["plan", "implement", "review", "handoff"], default="plan")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--state", action="append", default=[])
    parser.add_argument("--decision", action="append", default=[])
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--next-action", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--packet-json", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--start-timeout", type=float, default=180.0)
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    args = parser.parse_args(argv)
    if not (args.task or args.prompt_file or args.packet_json):
        parser.error("provide --task, --prompt-file, or --packet-json")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_turn(args)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
