#!/usr/bin/env python3
"""
benchmark_reasoning.py - fair benchmark for REASONING models on the
four competencies the lab assistant needs: physics, numerical analysis,
data handling, and data analysis.

Fairness rules (this is what makes it valid for R1-style models):
  * Generous token budget: reasoning models burn tokens thinking.
    Auto-detected by folder name (r1/deepscaler/reasoning/qwq/think) ->
    4096 tokens and no think-suppression; plain models get 1024.
  * Answer extraction: every prompt demands a final line
    'ANSWER: <value>'. We parse that line; if a model ignores the
    format, we fall back to the LAST number in the reply (reasoning
    traces put intermediate numbers FIRST, so last is the fair pick).
  * Per-question wall-clock is reported but NOT scored - correctness
    and speed are separate columns so slow-but-right is visible.

Families:
  P  physics            (4) - RLC damping, energy, induced EMF, skin time
  N  numerical analysis (3) - convergence order, Newton step, small-N RMS
  DH data handling      (3) - preamble detection, NaN audit, window size
  DA data analysis      (3) - saturation spotting, correlation sign,
                              baseline-window choice (lettered choice)

Usage:
  python3 scripts/benchmark_reasoning.py --backend mlx --models \
      "/Volumes/ScopeStudioModels/mlx/DeepSeek-R1-Distill-Qwen-7B-4bit,..."
  python3 scripts/benchmark_reasoning.py --mock
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

REASONING_HINT = re.compile(r"r1|deepscaler|reasoning|qwq|think", re.I)
FOOTER = "\nEnd your reply with a final line: ANSWER: <value>"


def T(name, prompt, truth, kind="num", tol=0.05):
    return dict(name=name, prompt=prompt + FOOTER, truth=truth,
                kind=kind, tol=tol)


P_TASKS = [
    T("damping ratio",
      "A series RLC discharge loop: R = 68 mOhm, L = 160 uH, C = 2.24 F. "
      "The damping ratio is zeta = (R/2) * sqrt(C/L). Compute zeta.",
      (0.068 / 2) * np.sqrt(2.24 / 160e-6)),                 # ~4.02
    T("coil energy",
      "A coil of inductance 160 uH carries 6620 A. Compute the stored "
      "magnetic energy E = 0.5*L*I^2 in joules.",
      0.5 * 160e-6 * 6620**2),                               # ~3506 J
    T("induced EMF",
      "Current through a 160 uH coil rises linearly from 0 to 6600 A "
      "in 4 ms. Compute the magnitude of the induced EMF V = L*dI/dt "
      "in volts.", 160e-6 * 6600 / 4e-3),                    # 264 V
    T("bank voltage drop",
      "A 2.24 F capacitor bank delivers 25.4 coulombs during a pulse. "
      "Compute the bank voltage drop dV = Q/C in volts.",
      25.4 / 2.24),                                          # ~11.34 V
]

N_TASKS = [
    T("convergence order",
      "The composite trapezoid rule has error proportional to h^2. If "
      "the step size h is halved, by what factor does the error "
      "decrease?", 4.0),
    T("Newton step",
      "One Newton-Raphson step for f(x) = x^2 - 5 starting at x0 = 2: "
      "x1 = x0 - f(x0)/f'(x0). Compute x1.", 2.25, tol=0.01),
    T("small-N RMS",
      "Compute the RMS (root mean square) of these five values: "
      "3, 4, 0, -4, -3.", np.sqrt((9 + 16 + 0 + 16 + 9) / 5),  # ~3.1623
      tol=0.02),
]

DH_TASKS = [
    T("preamble detection",
      "An oscilloscope CSV begins with these lines (numbered from 0):\n"
      "0: Model,DPO2024B\n1: Sample Interval,1.6e-07\n"
      "2: Vertical Units,V,,A\n3: TIME,CH1,CH2\n"
      "4: -1.96e-02,2.54,60.0\n5: -1.9598e-02,2.54,60.1\n"
      "At which line number does numeric DATA start?", 4.0,
      kind="int"),
    T("NaN audit",
      "A column contains: 2.5, NaN, 2.6, 2.4, NaN, NaN, 2.5, 2.6. "
      "How many values are missing (NaN)?", 3.0, kind="int"),
    T("window sample count",
      "Sample interval dt = 1.6e-7 s. How many samples fall in a "
      "5-millisecond window (window/dt, ignore endpoint effects)?",
      31250.0, tol=0.01),
]

DA_TASKS = [
    T("spot the saturated channel",
      "Stats over a shot - CH1: peak 6670 A, RMS 5192 A, crest factor "
      "1.28, top of waveform flat for 35 ms within 0.5% of max. "
      "CH2: peak 6586 A, RMS 954 A, crest factor 6.9, sharp 5 ms "
      "pulse. Which channel shows range saturation: A) CH1  B) CH2  "
      "C) both  D) neither. Answer with the letter.",
      "A", kind="choice"),
    T("correlation sign",
      "Across 14 shots, charging voltage rises monotonically and the "
      "measured droop time constant falls monotonically. The Pearson "
      "correlation between them is: A) positive  B) negative  "
      "C) zero  D) undefined. Answer with the letter.",
      "B", kind="choice"),
    T("baseline window choice",
      "To remove a constant sensor offset from a pulse measurement, "
      "which data window should define the baseline: A) the pulse "
      "plateau  B) the pre-trigger region before the pulse  C) the "
      "falling edge  D) the full record. Answer with the letter.",
      "B", kind="choice"),
]

FAMILIES = [("P", P_TASKS), ("N", N_TASKS), ("DH", DH_TASKS),
            ("DA", DA_TASKS)]

ANS_RE = re.compile(r"ANSWER\s*[:=]\s*([^\n]+)", re.I)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", re.I)


def extract(reply: str, kind: str):
    reply = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
    m = ANS_RE.search(reply)
    tail = m.group(1).strip() if m else reply.strip()
    if kind == "choice":
        lm = re.search(r"\b([ABCD])\b", tail.upper())
        if lm:
            return lm.group(1)
        lm = re.findall(r"\b([ABCD])\b", reply.upper())
        return lm[-1] if lm else None
    nums = NUM_RE.findall(tail.replace(",", ""))
    if not nums:
        nums = NUM_RE.findall(reply.replace(",", ""))
        return float(nums[-1]) if nums else None     # LAST number = fair
    return float(nums[0])


def score(task, reply):
    val = extract(reply, task["kind"])
    if val is None:
        return False, "no answer found"
    if task["kind"] == "choice":
        return val == task["truth"], f"{val}"
    if task["kind"] == "int":
        return float(val) == float(task["truth"]), f"{val:g}"
    rel = abs(val - task["truth"]) / max(abs(task["truth"]), 1e-12)
    return rel <= task["tol"], f"{val:g} (truth {task['truth']:.4g})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mlx")
    ap.add_argument("--models", default="")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "backtests", "reasoning_benchmark.txt"))
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
    lines, summary = [], {}
    total = sum(len(t) for _f, t in FAMILIES)
    for mdl in models:
        name = os.path.basename(str(mdl).rstrip("/"))
        is_reasoner = bool(REASONING_HINT.search(name))
        max_tok = 4096 if is_reasoner else 1024
        n_ok, t_sum = 0, 0.0
        for fam, tasks in FAMILIES:
            for t in tasks:
                t0 = time.perf_counter()
                if mdl == "mock:perfect":
                    reply = f"ANSWER: {t['truth']}"
                elif mdl == "mock:naive":
                    reply = "thinking... ANSWER: Z" \
                        if t["kind"] == "choice" else "ANSWER: -1"
                else:
                    reply = ask_model(
                        t["prompt"], model=mdl, backend=args.backend,
                        system_prompt="You are a careful physicist and "
                        "data analyst. Reason as needed, then give the "
                        "final line exactly as requested.",
                        max_tokens=max_tok)
                el = time.perf_counter() - t0
                t_sum += el
                ok, detail = score(t, reply)
                n_ok += ok
                lines.append(f"[{'PASS' if ok else 'FAIL'}] {name:34s} "
                             f"{fam:2s} {t['name']:26s} {detail} "
                             f"({el:.1f}s)")
        tag = " (reasoning budget)" if is_reasoner else ""
        summary[name] = (n_ok, t_sum / total, tag)
    hdr = ["Reasoning benchmark: physics / numerical / data handling / "
           f"data analysis (backend={args.backend})", ""]
    for k, (n, avg, tag) in summary.items():
        hdr.append(f"  {k:36s} {n}/{total}   avg {avg:5.1f}s{tag}")
    hdr.append("")
    text = "\n".join(hdr + lines)
    print(text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
