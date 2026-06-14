#!/usr/bin/env python3
"""
orchestrate_tool_dev.py - two specialized local AIs developing a tool
together, under a deterministic orchestrator.

Roles:
  CODER   (e.g. Qwen3-Coder / Qwen2.5-Coder) - drafts the implementation.
  ANALYST (math/science model, e.g. Qwen3-14B / Qwen3.5) - reviews the
          draft together with its measured test results and proposes a
          numerically better or cleaner revision.
  ORCHESTRATOR (this script - plain Python, not a model) - runs the
          screened, sandboxed tests on every candidate, picks the winner
          BY MEASUREMENT, and files it as an INACTIVE draft in
          tool_sandbox/drafts/ for human review. Models never decide
          what ships; tests and the human do.

Built-in demo task: the MAD despike algorithm (same spec as benchmark
family H), tested on a synthetic pulse with injected spikes against a
reference implementation.

Usage (Mac):
  python3 scripts/orchestrate_tool_dev.py --backend mlx \
      --coder  ~/models/mlx/Qwen2.5-Coder-14B-Instruct-4bit \
      --analyst ~/models/mlx/Qwen3-14B-4bit
  python3 scripts/orchestrate_tool_dev.py --mock     # validates the loop
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.benchmark_tool_creation import (   # noqa: E402
    BANNED, _extract_code, run_generated_tool, _ref_despike)

SPEC = (
    "def tool(y, w, k):\n"
    "    # 1) base = centered moving average of y, window w "
    "(np.convolve, mode='same')\n"
    "    # 2) r = y - base\n"
    "    # 3) sigma = 1.4826 * median(|r - median(r)|)   (MAD)\n"
    "    # 4) return a copy of y where samples with |r| > k*sigma are "
    "replaced by base at those samples\n"
)

CODER_PROMPT = (
    "Write a Python function using ONLY numpy (imported as np), "
    "implementing exactly this specification:\n\n" + SPEC +
    "\nReply with ONLY the complete function in a python code block.")

ANALYST_PROMPT = (
    "You are an applied mathematician reviewing a numerical tool.\n"
    "Specification:\n" + SPEC +
    "\nCandidate implementation:\n```python\n{code}\n```\n"
    "Measured result vs a reference implementation: {result}\n\n"
    "Improve it: fix any numerical error first; if already correct, make "
    "it more robust (edge windows, w<=1, empty arrays) WITHOUT changing "
    "the specified behavior on valid input. Use ONLY numpy. Reply with "
    "ONLY the complete improved function in a python code block.")


def make_test_signal():
    rng = np.random.default_rng(7)
    y = np.where(np.arange(8000) > 1500, 5000.0, 0.0) + \
        rng.normal(0, 20, 8000)
    y[np.array([2200, 3300, 4400, 5500])] += 2500.0
    return y


def evaluate(code: str, y: np.ndarray, args: dict):
    """Returns (score or None, detail). Score = NRMSE vs reference
    (lower is better); None = failed screen/execution."""
    out, err = run_generated_tool(code, y, args)
    if out is None:
        return None, err
    ref = _ref_despike(y, args["w"], args["k"])
    if np.asarray(out).shape != ref.shape:
        return None, "shape mismatch"
    nrmse = float(np.sqrt(np.nanmean((np.asarray(out) - ref) ** 2))
                  / max(float(np.std(ref)), 1e-12))
    return nrmse, f"NRMSE {nrmse:.5f}"


def ask(backend: str, model: str, prompt: str) -> str:
    from ai_assistant import ask_model
    return ask_model(prompt, model=model, backend=backend,
                     system_prompt="You are a precise engineering "
                     "assistant. Output format exactly as requested.",
                     max_tokens=900)


MOCK_CODER = ("```python\ndef tool(y, w, k):\n"
              "    kern = np.ones(int(w))/int(w)\n"
              "    base = np.convolve(y, kern, mode='same')\n"
              "    r = y - base\n"
              "    sig = 1.4826*np.median(np.abs(r))\n"     # subtle bug:
              "    out = y.copy()\n"                         # no median(r)
              "    out[np.abs(r) > k*sig] = base[np.abs(r) > k*sig]\n"
              "    return out\n```")
MOCK_ANALYST = ("```python\ndef tool(y, w, k):\n"
                "    w = max(int(w), 1)\n"
                "    kern = np.ones(w)/w\n"
                "    base = np.convolve(y, kern, mode='same')\n"
                "    r = y - base\n"
                "    sig = 1.4826*np.median(np.abs(r - np.median(r)))\n"
                "    out = y.copy()\n"
                "    bad = np.abs(r) > k*sig\n"
                "    out[bad] = base[bad]\n"
                "    return out\n```")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mlx")
    ap.add_argument("--coder", default="")
    ap.add_argument("--analyst", default="")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    y = make_test_signal()
    targs = {"w": 31, "k": 5.0}
    log: list[str] = []

    def step(role, model, prompt, mock_reply):
        t0 = time.perf_counter()
        reply = mock_reply if args.mock else ask(
            args.backend, os.path.expanduser(model), prompt)
        code = _extract_code(reply or "")
        el = time.perf_counter() - t0
        if code is None:
            log.append(f"[{role}] no function produced ({el:.1f}s)")
            return None
        score, detail = evaluate(code, y, targs)
        log.append(f"[{role}] {detail} ({el:.1f}s)")
        return {"role": role, "code": code, "score": score,
                "detail": detail}

    if not args.mock and (not args.coder or not args.analyst):
        print("Give --coder and --analyst model paths (or --mock).")
        return 2

    draft = step("CODER", args.coder, CODER_PROMPT, MOCK_CODER)
    candidates = [c for c in [draft] if c and c["score"] is not None]

    if draft and draft["code"]:
        rev = step("ANALYST", args.analyst,
                   ANALYST_PROMPT.format(code=draft["code"],
                                         result=draft["detail"]),
                   MOCK_ANALYST)
        if rev and rev["score"] is not None:
            candidates.append(rev)

    print("\n".join(log))
    if not candidates:
        print("\nNo candidate passed the sandbox - nothing filed.")
        return 1
    winner = min(candidates, key=lambda c: c["score"])
    print(f"\nWINNER: {winner['role']} ({winner['detail']})")

    # file as an INACTIVE draft for human review (never auto-promoted)
    from tool_sandbox import create_draft_tool
    try:
        paths = create_draft_tool(
            "mad_despike_orchestrated",
            purpose=("MAD despike drafted by CODER, revised by ANALYST, "
                     "winner selected by measured NRMSE. "
                     f"{winner['role']}: {winner['detail']}. "
                     "Review + promote manually."))
        with open(paths.script, "w") as fh:
            fh.write("import numpy as np\n\n" + winner["code"] + "\n")
        print(f"Draft filed (INACTIVE): {paths.folder}")
        print("Review it, run its test, then promote manually per "
              "tool_sandbox/README.md.")
    except Exception as e:
        print(f"(draft filing skipped: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
