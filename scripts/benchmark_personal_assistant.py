#!/usr/bin/env python3
"""
benchmark_personal_assistant.py - selecting the resident problem-solver.

Tests the three faculties the user wants in a personal lab assistant:
sharp problem solving, pattern identification, and logical deduction -
in environments designed to be NEW (invented notation, novel framings of
Erdos-flavored results) so memorization doesn't help, plus measurable
algorithm optimization.

Families:
  L  logic & deduction in invented environments (3) - a novel iterated
     map (truth computed by simulation, not trusted to memory), sensor
     deduction, a uniqueness-checked mini constraint puzzle.
  R  Erdos-flavored combinatorics, SOLVED results, novel phrasing (3) -
     Ramsey R(3,3), Erdos-Szekeres, the happy-ending problem.
  S  Erdos-Straus SEARCH (1) - find x,y,z with 4/7 = 1/x+1/y+1/z.
     Graded by exact Fraction arithmetic: ANY valid triple passes, so
     this rewards search, not recall.
  O  algorithm optimization (1) - given a deliberately O(n*w) Python
     double loop, produce an equivalent that is MEASURED >= 10x faster
     in the sandbox (correctness via NRMSE + timed in-subprocess).
  A  applied numerical analysis (3) - streaming variance algorithm
     choice, error amplification of x^3 (condition number), stable
     least-squares choice.
  U  UNGRADED probe (1) - plan of attack for computationally probing an
     open conjecture; the reply is logged verbatim for HUMAN review of
     reasoning structure. Counts for nothing.

Fairness: same rules as benchmark_reasoning.py (ANSWER: line, last-
number fallback, 4096-token budget for reasoning-named models, time
reported but never scored).

Usage:
  python3 scripts/benchmark_personal_assistant.py --backend mlx \
      --models "<folder1>,<folder2>,..."
  python3 scripts/benchmark_personal_assistant.py --mock
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.benchmark_tool_creation import (   # noqa: E402
    BANNED, _extract_code)

REASONING_HINT = re.compile(r"r1|deepscaler|reasoning|qwq|think", re.I)
FOOTER = "\nEnd your reply with a final line: ANSWER: <value>"
ANS_RE = re.compile(r"ANSWER\s*[:=]\s*([^\n]+)", re.I)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", re.I)


# ---- L1 truth computed by simulation, never trusted to memory ----------
def _l1_steps(n: int = 5) -> int:
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 5 * n - 1
        steps += 1
        if steps > 10000:
            raise RuntimeError("map does not terminate")
    return steps


L_TASKS = [
    dict(name="invented iterated map",
         prompt=("A number machine applies one rule per step: if the "
                 "number is even, divide it by 2; if odd, replace n with "
                 "5n - 1. Starting from 5, how many steps until it first "
                 "reaches 1?"),
         truth=float(_l1_steps(5)), kind="int"),
    dict(name="sensor deduction",
         prompt=("Three current sensors measured a calibrated 100 A "
                 "source. Exactly one always over-reads, one always "
                 "under-reads, one is exact. P read 103, Q read 100, "
                 "R read 97. Which sensor is exact: A) P  B) Q  C) R? "
                 "Answer with the letter."),
         truth="B", kind="choice"),
    dict(name="rack constraint puzzle",
         prompt=("Three modules X, Y, Z occupy racks 1, 2, 3 (left to "
                 "right), one module per rack. Facts: X is immediately "
                 "left of Y. Z is not adjacent to X. Which module is in "
                 "rack 3: A) X  B) Y  C) Z? Answer with the letter."),
         truth="C", kind="choice"),
]

R_TASKS = [
    dict(name="monochromatic triangle",
         prompt=("In a group of people, every pair is either allied or "
                 "rivals. What is the SMALLEST group size that "
                 "guarantees three mutual allies or three mutual "
                 "rivals, no matter how relations are assigned?"),
         truth=6.0, kind="int"),
    dict(name="guaranteed monotone run",
         prompt=("Any sequence of 10 distinct real numbers must contain "
                 "an increasing or a decreasing subsequence of at least "
                 "what length? (Subsequence need not be contiguous.)"),
         truth=4.0, kind="int"),
    dict(name="convex quadrilateral",
         prompt=("What is the minimum number of points in the plane, no "
                 "three collinear, that guarantees some four of them "
                 "form a convex quadrilateral?"),
         truth=5.0, kind="int"),
]

S_TASK = dict(
    name="unit-fraction search 4/7",
    prompt=("Find positive integers x <= y <= z such that "
            "4/7 = 1/x + 1/y + 1/z. The three integers need not be "
            "distinct. Reply with the final line: "
            "ANSWER: x, y, z"))

A_TASKS = [
    dict(name="streaming variance",
         prompt=("You must compute the variance of a stream of 10^9 "
                 "samples in one pass without storing them. Which "
                 "algorithm: A) two-pass mean-then-variance  "
                 "B) Welford's online algorithm  C) sum of squares "
                 "minus square of sums in float32  D) sort first. "
                 "Answer with the letter."),
         truth="B", kind="choice"),
    dict(name="error amplification",
         prompt=("y = x^3 is evaluated at x = 10 with a 1% relative "
                 "error in x. To first order, what is the relative "
                 "error of y in percent?"),
         truth=3.0, kind="num", tol=0.05),
    dict(name="ill-conditioned least squares",
         prompt=("A least-squares design matrix has condition number "
                 "1e8 in double precision. Which solution method is "
                 "most numerically trustworthy: A) normal equations "
                 "(A^T A x = A^T b)  B) QR or SVD factorization  "
                 "C) matrix inversion  D) Cramer's rule. Answer with "
                 "the letter."),
         truth="B", kind="choice"),
]

U_TASK = dict(
    name="open-conjecture attack plan (UNGRADED)",
    prompt=("The Erdos-Straus conjecture (4/n = 1/x + 1/y + 1/z has "
            "positive integer solutions for every n >= 2) is open in "
            "general. WITHOUT attempting to prove it, outline in at "
            "most 8 numbered steps how you would computationally verify "
            "it for all n < 10^4, including how you would bound the "
            "search space and avoid floating-point pitfalls."))

O_BASELINE = (
    "def baseline(y, w):\n"
    "    out = [0.0]*len(y)\n"
    "    h = w//2\n"
    "    for i in range(len(y)):\n"
    "        s = 0.0; c = 0\n"
    "        for j in range(max(0, i-h), min(len(y), i+h+1)):\n"
    "            s += y[j]; c += 1\n"
    "        out[i] = s/c\n"
    "    return out\n")

O_TASK = dict(
    name="optimize moving average",
    prompt=("This Python moving-average is correct but O(n*w):\n\n"
            "```python\n" + O_BASELINE + "```\n"
            "Write a function `def tool(y, w):` using ONLY numpy "
            "(imported as np) that returns the SAME values (partial "
            "windows at the edges, same length) but at least 10x "
            "faster for n = 20000, w = 101. Hint: a cumulative-sum or "
            "convolution approach works. Reply with ONLY the complete "
            "function in a python code block."))


def run_optimized(code: str, timeout: float = 60.0):
    """Sandbox-run model code vs the baseline IN THE SAME subprocess:
    returns (nrmse, speedup) or (None, reason)."""
    if BANNED.search(code):
        return None, "banned construct"
    with tempfile.TemporaryDirectory() as td:
        prog = os.path.join(td, "race.py")
        with open(prog, "w") as fh:
            fh.write(
                "import numpy as np, json, time\n"
                + O_BASELINE + "\n" + code + "\n"
                "rng = np.random.default_rng(0)\n"
                "y = rng.normal(0, 1, 20000)\n"
                "w = 101\n"
                "t0 = time.perf_counter(); ref = baseline(list(y), w); "
                "tb = time.perf_counter() - t0\n"
                "t0 = time.perf_counter(); out = tool(y, w); "
                "tm = time.perf_counter() - t0\n"
                "ref = np.asarray(ref); out = np.asarray(out, dtype=float)\n"
                "nrmse = float(np.sqrt(np.mean((out-ref)**2))/np.std(ref)) "
                "if out.shape == ref.shape else 1e9\n"
                "print(json.dumps({'nrmse': nrmse, "
                "'speedup': tb/max(tm, 1e-9)}))\n")
        try:
            r = subprocess.run([sys.executable, "-I", prog],
                               capture_output=True, timeout=timeout,
                               env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return None, "timeout"
        if r.returncode != 0:
            return None, (r.stderr.decode()[-160:] or "nonzero exit")
        try:
            d = json.loads(r.stdout.decode().strip().splitlines()[-1])
            return d, None
        except Exception as e:
            return None, f"no result: {e}"


def extract(reply: str, kind: str):
    reply = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
    m = ANS_RE.search(reply)
    tail = m.group(1).strip() if m else reply.strip()
    if kind == "choice":
        lm = re.search(r"\b([ABCD])\b", tail.upper()) or \
            (re.findall(r"\b([ABCD])\b", reply.upper()) or [None])
        return lm.group(1) if hasattr(lm, "group") else (
            lm[-1] if isinstance(lm, list) and lm else None)
    nums = NUM_RE.findall(tail.replace(",", " "))
    if not nums:
        nums = NUM_RE.findall(reply.replace(",", " "))
        return float(nums[-1]) if nums else None
    return float(nums[0])


def score_plain(task, reply):
    val = extract(reply, task["kind"])
    if val is None:
        return False, "no answer"
    if task["kind"] == "choice":
        return val == task["truth"], str(val)
    if task["kind"] == "int":
        return float(val) == float(task["truth"]), f"{val:g}"
    rel = abs(val - task["truth"]) / max(abs(task["truth"]), 1e-12)
    return rel <= task.get("tol", 0.05), f"{val:g}"


def score_straus(reply):
    reply = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
    m = ANS_RE.search(reply)
    src = m.group(1) if m else reply
    ints = re.findall(r"\d+", src)
    if len(ints) < 3:
        return False, "need three integers"
    x, y, z = (int(v) for v in ints[-3:])
    if min(x, y, z) <= 0:
        return False, f"{x},{y},{z}: non-positive"
    ok = Fraction(1, x) + Fraction(1, y) + Fraction(1, z) == Fraction(4, 7)
    return ok, f"1/{x}+1/{y}+1/{z}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mlx")
    ap.add_argument("--models", default="")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "backtests", "personal_assistant_benchmark.txt"))
    args = ap.parse_args()

    if args.mock:
        models = ["mock:perfect", "mock:naive"]
    else:
        models = [os.path.expanduser(m.strip())
                  for m in args.models.split(",") if m.strip()]
        if not models:
            print("Give --models (or --mock).")
            return 2

    from ai_assistant import ask_model
    graded = [("L", L_TASKS), ("R", R_TASKS), ("A", A_TASKS)]
    total = sum(len(t) for _f, t in graded) + 2      # + S + O
    lines, summary, probes = [], {}, []

    def ask(mdl, prompt, max_tok):
        if mdl == "mock:perfect":
            return None                              # handled per-task
        if mdl == "mock:naive":
            return "ANSWER: 999"
        return ask_model(prompt, model=mdl, backend=args.backend,
                         system_prompt="You are a sharp, careful problem "
                         "solver. Think as needed; end exactly as the "
                         "task requests.", max_tokens=max_tok)

    for mdl in models:
        name = os.path.basename(str(mdl).rstrip("/"))
        max_tok = 4096 if REASONING_HINT.search(name) else 1536
        n_ok, t_sum, n_t = 0, 0.0, 0

        def run_one(fam, tname, prompt, scorer, perfect_reply):
            nonlocal n_ok, t_sum, n_t
            t0 = time.perf_counter()
            reply = ask(mdl, prompt, max_tok)
            if reply is None:
                reply = perfect_reply
            el = time.perf_counter() - t0
            t_sum += el
            n_t += 1
            ok, detail = scorer(reply)
            n_ok += ok
            lines.append(f"[{'PASS' if ok else 'FAIL'}] {name:30s} "
                         f"{fam:2s} {tname:28s} {detail} ({el:.1f}s)")

        for fam, tasks in graded:
            for t in tasks:
                run_one(fam, t["name"], t["prompt"] + FOOTER,
                        lambda r, t=t: score_plain(t, r),
                        f"ANSWER: {t['truth']}")
        run_one("S", S_TASK["name"], S_TASK["prompt"], score_straus,
                "ANSWER: 2, 15, 210")
        run_one("O", O_TASK["name"], O_TASK["prompt"],
                lambda r: (lambda c: (False, "no function") if c is None
                           else (lambda d, e: (False, e) if d is None else
                                 (d["nrmse"] < 0.02 and d["speedup"] >= 10,
                                  f"NRMSE {d['nrmse']:.4f}, "
                                  f"{d['speedup']:.0f}x"))(
                                     *run_optimized(c)))(_extract_code(r)),
                "```python\ndef tool(y, w):\n"
                "    c = np.cumsum(np.concatenate(([0.0], y)))\n"
                "    n = len(y); h = w//2\n"
                "    i = np.arange(n)\n"
                "    lo = np.maximum(0, i-h); hi = np.minimum(n, i+h+1)\n"
                "    return (c[hi]-c[lo])/(hi-lo)\n```")
        # ungraded probe - logged verbatim for human review
        t0 = time.perf_counter()
        probe = ask(mdl, U_TASK["prompt"], max_tok) or "(mock)"
        probes.append(f"--- {name} | {U_TASK['name']} "
                      f"({time.perf_counter()-t0:.0f}s) ---\n{probe}\n")
        summary[name] = (n_ok, t_sum / max(n_t, 1))

    hdr = ["Personal-assistant benchmark: logic / Erdos-flavored "
           f"deduction / search / optimization / numerics "
           f"(backend={args.backend})", ""]
    for k, (n, avg) in summary.items():
        hdr.append(f"  {k:32s} {n}/{total}   avg {avg:5.1f}s")
    hdr.append("")
    text = "\n".join(hdr + lines)
    text += ("\n\n==== UNGRADED open-conjecture probes "
             "(judge the reasoning yourself) ====\n\n"
             + "\n".join(probes))
    print(text[:6000])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(text + "\n")
    print(f"\nfull report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
