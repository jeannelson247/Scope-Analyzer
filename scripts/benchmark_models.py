#!/usr/bin/env python3
"""
Benchmark local model profiles for Scope Studio.

This script is intentionally small. It checks whether a model can return a
valid plot-action JSON block, which is more useful for Scope Studio than a
generic chatbot benchmark.
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ai_assistant import ask_model  # noqa: E402
from chat_actions import extract_actions  # noqa: E402
from model_catalog import MODEL_PROFILES  # noqa: E402


PROMPT = """
You are controlling Scope Studio. Change the plot to a Nature Physics style
figure with Time (ms) on the x-axis, Current (A) on the left axis, Control
Signal (V) on the right axis, and a 10 kHz low-pass filter for current-like
channels. End with only the JSON action block.
"""


SYSTEM = """
You are a plot-action router. Reply with one fenced JSON block only.
Allowed actions include set_journal_style, set_xlabel, set_ylabel_left,
set_ylabel_right, lowpass_filter, and set_plot_style.
"""


def run_profile(profile) -> dict:
    start = time.perf_counter()
    text = ask_model(
        PROMPT,
        model=profile.model,
        backend=profile.backend,
        system_prompt=SYSTEM,
        max_tokens=256,
    )
    elapsed = time.perf_counter() - start
    _, actions = extract_actions(text)
    action_names = set()
    for action in actions:
        action_names.update(action)
    return {
        "profile": profile.name,
        "tier": profile.tier,
        "backend": profile.backend,
        "model": profile.model,
        "seconds": round(elapsed, 3),
        "valid_actions": bool(actions),
        "action_names": sorted(action_names),
        "raw_preview": text[:500],
    }


def main():
    results = []
    for profile in MODEL_PROFILES:
        print(f"Benchmarking {profile.name}...")
        results.append(run_profile(profile))
    out_path = os.path.join(ROOT, "backtests", "model_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

