#!/usr/bin/env python3
"""
Benchmark local MLX models for Scope Studio 03.

This is not a generic chatbot benchmark. It scores the way the app will use
the assistant:

1. Plot/action routing with reversible in-RAM display changes.
2. NumPy-first data-analysis/code drafting.
3. Power-electronics/current-monitor physics reasoning.
4. Scope-data interpretation with explicit guardrails.

Each model is run in a separate Python subprocess so large MLX weights are
released before the next model loads.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_assistant import mlx_model_roots

REPORT_DIR = ROOT / "backtests"
RESULT_JSON = REPORT_DIR / "mlx_model_benchmark.json"
RESULT_TXT = REPORT_DIR / "mlx_model_benchmark_report.txt"


SYSTEM = (
    "You are the local Scope Studio assistant for oscilloscope CSV plotting, "
    "current-monitor calibration, and educational publication figures. "
    "Hard rule: original CSV files are immutable. You may only propose "
    "reversible in-memory transforms, overlays, exported copies, or "
    "deterministic NumPy/SciPy/MLX tools."
)


TASKS = [
    {
        "key": "tool_routing",
        "weight": 0.30,
        "max_tokens": 220,
        "prompt": (
            "The user says: make this a Nature Physics style plot, label x as "
            "Time (ms), left y as Current (A), right y as Control Signal (V), "
            "align the y-axis zeros, and apply a 10 kHz low-pass filter to "
            "current traces. Allowed action keys include set_journal_style, "
            "set_xlabel, set_ylabel_left, set_ylabel_right, align_zeros, "
            "lowpass_filter, and set_plot_style. Each action must be a JSON "
            "object, not a string. Example: {\"set_xlabel\": \"Time (ms)\"}. "
            "Reply with exactly one fenced JSON block:\n"
            "```json\n{\"actions\": [...]}\n```"
        ),
    },
    {
        "key": "coding_data_analysis",
        "weight": 0.25,
        "max_tokens": 360,
        "prompt": (
            "Draft a compact Python function using NumPy that computes dI/dt "
            "from time and current arrays and estimates inductive voltage as "
            "V_L = L*dI/dt for L=166e-6 H. It must not write files or modify "
            "the input arrays. Include only the code block."
        ),
    },
    {
        "key": "physics_reasoning",
        "weight": 0.30,
        "max_tokens": 360,
        "prompt": (
            "A tokamak TF coil driver shot uses a BBCM and a Pearson current "
            "monitor. The BBCM saturates near 6.6 kA. Pearson is trusted for "
            "the first 5 ms, and the BBCM is trusted again after 40 ms. The "
            "internal inductance is 166 uH. Explain a safe workflow to "
            "estimate the missing current interval and inductive voltage. "
            "State limitations and keep measured data separate from overlays."
        ),
    },
    {
        "key": "scope_interpretation",
        "weight": 0.15,
        "max_tokens": 300,
        "prompt": (
            "Visible-window stats: CH1 BBCM calibrated current peak=6600 A, "
            "mean=4100 A, RMS=4300 A; CH2 Pearson peak=1450 A in first 5 ms; "
            "control signal goes 0 to 8 V from 5 ms to 80 ms; low-pass "
            "cutoff=15 kHz. Give a concise lab-assistant interpretation and "
            "what deterministic tool you would run next."
        ),
    },
]


def folder_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def gb(nbytes: int) -> float:
    return nbytes / 1024**3


def is_complete_model(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    if any(path.rglob("*.incomplete")):
        return False
    return any(path.glob("*.safetensors"))


def model_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for root in mlx_model_roots():
        path = Path(root).expanduser()
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            roots.append(path)
            seen.add(key)
    return roots


def iter_model_dirs() -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for root in model_roots():
        if not root.exists():
            continue
        candidates = [root] if is_complete_model(root) else []
        if root.is_dir():
            candidates.extend(p for p in root.iterdir() if p.is_dir())
        for path in candidates:
            key = str(path.resolve())
            if key not in seen:
                out.append(path)
                seen.add(key)
    return sorted(out, key=lambda p: p.name)


def installed_models(max_gb: float) -> list[Path]:
    out = []
    for path in iter_model_dirs():
        size = gb(folder_size(path))
        if size <= max_gb and is_complete_model(path):
            out.append(path)
    return out


def _has(text: str, *needles: str) -> int:
    low = text.lower()
    return sum(1 for needle in needles if needle.lower() in low)


def score_tool_routing(text: str) -> tuple[float, list[str]]:
    notes = []
    score = 0.0
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if m:
        score += 2.0
        try:
            payload = json.loads(m.group(1))
            actions = payload.get("actions", [])
            if isinstance(actions, list) and actions:
                score += 2.0
                object_actions = [a for a in actions if isinstance(a, dict)]
                if not object_actions:
                    notes.append("actions list had no action objects")
                    return min(score, 10.0), notes
                flat = json.dumps(object_actions).lower()
                checks = [
                    ("set_xlabel", 1.0),
                    ("set_ylabel_left", 1.0),
                    ("set_ylabel_right", 1.0),
                    ("align_zeros", 1.0),
                    ("lowpass_filter", 1.5),
                    ("10000", 0.8),
                    ("nature", 0.7),
                ]
                for key, points in checks:
                    if key in flat:
                        score += points
                    else:
                        notes.append(f"missing {key}")
            else:
                notes.append("JSON block had no actions list")
        except json.JSONDecodeError:
            notes.append("JSON block was malformed")
    else:
        notes.append("no fenced JSON action block")
    return min(score, 10.0), notes


def score_code(text: str) -> tuple[float, list[str]]:
    notes = []
    low = text.lower()
    score = 0.0
    checks = [
        ("def ", 1.5),
        ("numpy", 1.0),
        ("np.", 1.0),
        ("gradient", 1.5),
        ("166e-6", 1.0),
        ("l *", 1.0),
        ("return", 1.0),
        ("copy", 0.6),
        ("asarray", 0.6),
    ]
    for key, points in checks:
        if key in low:
            score += points
        else:
            notes.append(f"missing {key}")
    unsafe = ["open(", "to_csv", "savetxt", "write(", "inplace=true"]
    found_unsafe = [u for u in unsafe if u in low]
    if not found_unsafe:
        score += 1.8
    else:
        notes.append("unsafe write-like token: " + ", ".join(found_unsafe))
    return min(score, 10.0), notes


def score_physics(text: str) -> tuple[float, list[str]]:
    notes = []
    score = 0.0
    groups = [
        (["pearson"], 1.0, "Pearson reference"),
        (["bbcm", "busbar"], 1.0, "BBCM/busbar"),
        (["saturat", "censor", "clipp"], 1.2, "saturation/censoring"),
        (["5 ms", "first 5"], 1.0, "first 5 ms"),
        (["40 ms", "after 40"], 1.0, "after 40 ms"),
        (["166", "uh", "microhenry"], 1.0, "166 uH"),
        (["di/dt", "didt"], 1.0, "dI/dt"),
        (["l*d", "l * d", "inductive voltage"], 1.0, "L*dI/dt"),
        (["overlay", "estimate", "model"], 0.8, "overlay/model wording"),
        (["raw", "csv", "immutable", "do not modify", "untouched"], 1.0,
         "immutable CSV caveat"),
    ]
    low = text.lower()
    for needles, points, label in groups:
        if any(n in low for n in needles):
            score += points
        else:
            notes.append(f"missing {label}")
    return min(score, 10.0), notes


def score_interpretation(text: str) -> tuple[float, list[str]]:
    notes = []
    low = text.lower()
    score = 0.0
    checks = [
        ("6600", 1.0),
        ("saturat", 1.4),
        ("pearson", 1.0),
        ("15 khz", 1.0),
        ("low-pass", 0.8),
        ("control", 0.8),
        ("deterministic", 1.0),
        ("reconstruct", 1.0),
        ("overlay", 1.0),
        ("not measured", 1.0),
    ]
    for key, points in checks:
        if key in low:
            score += points
        else:
            notes.append(f"missing {key}")
    return min(score, 10.0), notes


SCORERS = {
    "tool_routing": score_tool_routing,
    "coding_data_analysis": score_code,
    "physics_reasoning": score_physics,
    "scope_interpretation": score_interpretation,
}


def parse_child_json(stdout: str) -> dict:
    """mlx-lm can print warnings before the JSON result for large models.
    Parse the last JSON object instead of requiring pristine stdout."""
    text = stdout.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    end = text.rfind("}")
    if end < 0:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    depth = 0
    in_str = False
    esc = False
    start = -1
    for i in range(end, -1, -1):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                start = i
                break
    if start < 0:
        raise json.JSONDecodeError("unterminated JSON object", text, end)
    return json.loads(text[start:end + 1])


def role_scores(task_results: list[dict], size_gb: float,
                avg_seconds: float) -> dict:
    scores = {r["task"]: float(r["score_10"]) for r in task_results}
    tool_coder = (
        0.55 * scores.get("tool_routing", 0.0)
        + 0.45 * scores.get("coding_data_analysis", 0.0)
    )
    analyst = (
        0.25 * scores.get("tool_routing", 0.0)
        + 0.40 * scores.get("physics_reasoning", 0.0)
        + 0.35 * scores.get("scope_interpretation", 0.0)
    )
    lightweight_default = (
        0.45 * scores.get("tool_routing", 0.0)
        + 0.25 * scores.get("coding_data_analysis", 0.0)
        + 0.15 * scores.get("physics_reasoning", 0.0)
        + 0.15 * scores.get("scope_interpretation", 0.0)
    )
    return {
        "tool_coder_score_10": round(tool_coder, 2),
        "tool_coder_pass": tool_coder >= 8.0,
        "analyst_score_10": round(analyst, 2),
        "analyst_pass": analyst >= 7.5,
        "lightweight_default_score_10": round(lightweight_default, 2),
        "lightweight_default_pass": (
            lightweight_default >= 8.0 and size_gb <= 4.5
            and avg_seconds <= 8.0
        ),
    }


def run_single(model_path: Path) -> dict:
    from ai_assistant import ask_model

    model_name = model_path.name
    task_results = []
    t0 = time.perf_counter()
    for task in TASKS:
        start = time.perf_counter()
        reply = ask_model(
            task["prompt"],
            model=str(model_path),
            backend="mlx",
            system_prompt=SYSTEM,
            max_tokens=int(task["max_tokens"]),
        )
        elapsed = time.perf_counter() - start
        if reply.startswith("AI backend error:"):
            score, notes = 0.0, [reply[:240]]
        else:
            scorer = SCORERS[task["key"]]
            score, notes = scorer(reply)
        task_results.append({
            "task": task["key"],
            "score_10": round(score, 2),
            "seconds": round(elapsed, 3),
            "notes": notes,
            "preview": reply[:900],
        })

    weighted = sum(
        r["score_10"] * next(t["weight"] for t in TASKS
                             if t["key"] == r["task"])
        for r in task_results
    )
    total_seconds = time.perf_counter() - t0
    avg_seconds = total_seconds / max(len(task_results), 1)

    speed_bonus = 0.0
    if avg_seconds <= 8:
        speed_bonus = 0.5
    elif avg_seconds > 45:
        speed_bonus = -0.5
    elif avg_seconds > 90:
        speed_bonus = -1.0

    size_gb = gb(folder_size(model_path))
    memory_penalty = -0.5 if size_gb > 15 else 0.0
    final = max(0.0, min(10.0, weighted + speed_bonus + memory_penalty))
    roles = role_scores(task_results, size_gb, avg_seconds)

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB. This app is Mac-first, but keep
    # a readable fallback.
    maxrss_gb = maxrss / 1024**3 if maxrss > 50_000_000 else maxrss / 1024**2

    return {
        "model": model_name,
        "path": str(model_path),
        "size_gb": round(size_gb, 2),
        "weighted_score_10": round(weighted, 2),
        "speed_bonus": speed_bonus,
        "memory_penalty": memory_penalty,
        "final_score_10": round(final, 2),
        **roles,
        "total_seconds": round(total_seconds, 3),
        "avg_seconds_per_task": round(avg_seconds, 3),
        "max_rss_gb": round(maxrss_gb, 2),
        "tasks": task_results,
    }


def write_reports(results: list[dict], skipped: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=lambda r: r["final_score_10"], reverse=True)
    payload = {
        "created_unix": time.time(),
        "machine_note": "Apple Silicon MLX local benchmark; original CSV untouched.",
        "results_ranked": ranked,
        "skipped": skipped,
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")

    lines = [
        "Scope Studio 03 MLX Model Benchmark",
        "=" * 38,
        "",
        "Purpose: rank local MLX models for Scope Studio assistant use, not "
        "generic chat.",
        "Guardrail: the benchmark prompts require immutable source CSVs and "
        "in-RAM/reversible transforms only.",
        "",
        "Ranking",
        "-------",
    ]
    for i, r in enumerate(ranked, 1):
        role_bits = [
            "tool/coder PASS" if r.get("tool_coder_pass") else "tool/coder fail",
            "analyst PASS" if r.get("analyst_pass") else "analyst fail",
            "light default PASS" if r.get("lightweight_default_pass")
            else "light default fail",
        ]
        lines.append(
            f"{i}. {r['model']} | final {r['final_score_10']}/10 | "
            f"size {r['size_gb']} GB | avg {r['avg_seconds_per_task']} s/task"
        )
        lines.append("   roles: " + "; ".join(role_bits))
        lines.append(
            f"   role scores: tool/coder {r.get('tool_coder_score_10')}/10, "
            f"analyst {r.get('analyst_score_10')}/10, "
            f"light default {r.get('lightweight_default_score_10')}/10"
        )
        lines.append(f"   path: {r['path']}")
        for task in r["tasks"]:
            lines.append(
                f"   - {task['task']}: {task['score_10']}/10, "
                f"{task['seconds']} s"
            )
            if task["notes"]:
                lines.append("     notes: " + "; ".join(task["notes"][:4]))
    if skipped:
        lines += ["", "Skipped", "-------"]
        for s in skipped:
            lines.append(f"- {s['model']}: {s['reason']}")
    RESULT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> int:
    candidates = installed_models(args.max_gb)
    skipped = []
    if args.include_incomplete:
        all_dirs = iter_model_dirs()
    else:
        all_dirs = candidates
    complete_names = {p.name for p in candidates}
    for p in iter_model_dirs():
        if not p.is_dir() or p.name in complete_names:
            continue
        size = gb(folder_size(p))
        reason = "incomplete download or missing safetensors"
        if size > args.max_gb:
            reason = f"over {args.max_gb:g} GB limit"
        skipped.append({"model": p.name, "path": str(p), "reason": reason,
                        "size_gb": round(size, 2)})

    results = []
    for path in all_dirs:
        if args.only and path.name not in args.only:
            continue
        if not is_complete_model(path):
            continue
        print(f"\n=== Benchmarking {path.name} ===", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--single", str(path)]
        proc = subprocess.run(cmd, text=True, capture_output=True,
                              timeout=args.timeout)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            skipped.append({"model": path.name, "path": str(path),
                            "reason": f"benchmark failed rc={proc.returncode}"})
            continue
        try:
            result = parse_child_json(proc.stdout)
        except json.JSONDecodeError:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            skipped.append({"model": path.name, "path": str(path),
                            "reason": "benchmark emitted non-JSON output"})
            continue
        results.append(result)
        print(
            f"{path.name}: {result['final_score_10']}/10, "
            f"{result['avg_seconds_per_task']} s/task",
            flush=True,
        )
    write_reports(results, skipped)
    print(f"\nWrote {RESULT_JSON}")
    print(f"Wrote {RESULT_TXT}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", type=Path,
                    help="internal: benchmark one model folder and print JSON")
    ap.add_argument("--max-gb", type=float, default=18.0)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--include-incomplete", action="store_true")
    ap.add_argument("--only", nargs="*",
                    help="limit to specific model folder names")
    args = ap.parse_args(argv)
    if args.single:
        print(json.dumps(run_single(args.single), indent=2))
        return 0
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
