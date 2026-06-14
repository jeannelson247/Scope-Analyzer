#!/usr/bin/env python3
"""
Benchmark Scope Studio's local model backends on the same short prompts.

Usage from the Scope Studio project root:
  python scripts/benchmark_mlx_direct.py --mlx ~/models/mlx/Qwen2.5-Coder-7B-Instruct-4bit
  python scripts/benchmark_mlx_direct.py --auto-mlx --ollama qwen3.5:9b-mlx,qwen3.5:9b

This is a speed/smoke benchmark, not a scientific-quality leaderboard.
It checks whether the model loads, answers grounded scope-analysis prompts,
and how long each backend takes after the first call.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_assistant import ask_model, list_mlx_models  # noqa: E402

PROMPTS = [
    (
        "tool_route",
        "User asks: scan the visible scope window for spikes above 5 sigma. "
        "Reply with the JSON action only."
    ),
    (
        "scope_interpret",
        "Visible stats: CH1 BBCM peak 6600 A, mean 4100 A, RMS 4300 A; "
        "CH2 Pearson peak 1450 A in first 5 ms; control signal rises 0-8 V. "
        "Give a concise lab-assistant interpretation and one deterministic "
        "tool to run next."
    ),
    (
        "coding_helper",
        "Write a small NumPy function estimate_inductive_voltage(time, current) "
        "that returns V_L = 166e-6 * dI/dt. Keep it short."
    ),
]


def split_csv(text: str) -> list[str]:
    return [x.strip() for x in (text or "").split(",") if x.strip()]


def bench_one(backend: str, model: str, max_tokens: int = 256):
    print(f"\n=== {backend} | {model} ===", flush=True)
    times = []
    for name, prompt in PROMPTS:
        t0 = time.perf_counter()
        out = ask_model(prompt, model=model, backend=backend, max_tokens=max_tokens)
        dt = time.perf_counter() - t0
        times.append(dt)
        preview = " ".join((out or "").split())[:180]
        print(f"{name:16s} {dt:8.3f} s | {len(out or ''):5d} chars | {preview}")
    cold = times[0]
    warm = sum(times[1:]) / max(len(times) - 1, 1)
    print(f"summary: cold_first={cold:.3f}s warm_avg={warm:.3f}s all_avg={sum(times)/len(times):.3f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx", default="", help="Comma-separated MLX repo IDs or local model folders.")
    ap.add_argument("--auto-mlx", action="store_true", help="Benchmark all local MLX model folders found under ~/models/mlx and ~/models.")
    ap.add_argument("--ollama", default="", help="Comma-separated Ollama tags to compare.")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    models: list[tuple[str, str]] = []
    for m in split_csv(args.mlx):
        models.append(("mlx", os.path.expanduser(m)))
    if args.auto_mlx:
        for m in list_mlx_models():
            models.append(("mlx", m))
    for m in split_csv(args.ollama):
        models.append(("ollama", m))

    if not models:
        print("No models supplied. Example:")
        print("  python scripts/benchmark_mlx_direct.py --auto-mlx")
        print("  python scripts/benchmark_mlx_direct.py --mlx mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
        return 2

    print("Scope Studio backend speed smoke test")
    print("Note: first MLX call includes model load time; later calls are the useful warm latency.")
    for backend, model in models:
        try:
            bench_one(backend, model, max_tokens=args.max_tokens)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"\n=== {backend} | {model} ===\nFAILED: {exc!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
