#!/usr/bin/env python3
"""
benchmark_tool_creation.py - Can a local model WRITE and CHOOSE tools?

Two families, scored against deterministic ground truth:

  D. tool-creation   - the model must write a small numpy-only function
                       (moving average; first-order low-pass). The
                       generated code is screened for banned tokens,
                       executed in an ISOLATED subprocess (python -I,
                       timeout, no inherited env) against a synthetic
                       signal, and scored by NRMSE vs the reference
                       implementation. This measures fitness for the
                       tool_sandbox draft workflow.
  E. algorithm-selection - scenario prompts (incl. the real 6.6 kA
                       de-censoring case: busbar soft-saturated, Pearson
                       core collapsing, true peak ~7.2 kA) where the
                       model must pick the right PRELOADED tool from the
                       registry. Tests judgment, not code.

Safety: generated code never runs in the app process. It is screened
(import whitelist: numpy only; no os/sys/open/exec/eval/__) and executed
via `python -I -c` in a subprocess with a 15 s timeout, exchanging data
through temp .npy files. This script is a developer benchmark - it is
not reachable from the Scope Studio UI.

Usage (Mac):
  python3 scripts/benchmark_tool_creation.py --backend mlx \
      --models "~/models/mlx/Qwen3-Coder-30B-A3B-Instruct-4bit,~/models/mlx/Qwen3.5-9B-4bit"
  python3 scripts/benchmark_tool_creation.py --backend ollama --models qwen3.5:9b-mlx
  python3 scripts/benchmark_tool_creation.py --mock
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

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BANNED = re.compile(
    r"\b(import\s+(?!numpy\b|math\b)\w+|from\s+(?!numpy\b|math\b)\w+|open\s*\(|exec|"
    r"eval|__\w+__|os\.|sys\.|subprocess|socket|pathlib|shutil|input\s*\()")

NRMSE_PASS = 0.05      # generated tool must be within 5% NRMSE of reference


# --------------------------------------------------------------------------
# Family D - tool creation tasks
# --------------------------------------------------------------------------
def _ref_movmean(y, w):
    k = np.ones(int(w)) / int(w)
    return np.convolve(y, k, mode="same")


def _ref_lowpass1(y, dt, fc):
    a = float(2 * np.pi * fc * dt / (2 * np.pi * fc * dt + 1))
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = out[i - 1] + a * (y[i] - out[i - 1])
    return out


D_TASKS = [
    dict(
        name="moving average",
        prompt=(
            "Write a Python function using ONLY numpy (already imported "
            "as np):\n\ndef tool(y, w):\n    # y: 1-D float array, w: odd "
            "window length (int)\n    # return the centered moving "
            "average of y with window w,\n    # same length as y "
            "(edges may use partial windows or zero padding)\n\n"
            "Reply with ONLY the complete function definition in a "
            "python code block, no explanation."),
        args={"w": 21},
        ref=lambda y, a: _ref_movmean(y, a["w"]),
        tol=0.08,        # edge conventions differ; be a bit lenient
    ),
    dict(
        name="first-order low-pass",
        prompt=(
            "Write a Python function using ONLY numpy (already imported "
            "as np):\n\ndef tool(y, dt, fc):\n    # y: 1-D float array, "
            "dt: sample interval in seconds, fc: cutoff in Hz\n    # "
            "return y filtered by a causal FIRST-ORDER (single-pole) "
            "low-pass\n    # filter with cutoff fc, e.g. the discrete "
            "RC filter\n    # out[i] = out[i-1] + alpha*(y[i]-out[i-1]) "
            "with alpha = 2*pi*fc*dt/(2*pi*fc*dt + 1)\n\n"
            "Reply with ONLY the complete function definition in a "
            "python code block, no explanation."),
        args={"dt": 1.6e-7, "fc": 10_000.0},
        ref=lambda y, a: _ref_lowpass1(y, a["dt"], a["fc"]),
        tol=NRMSE_PASS,
    ),
]


def _extract_code(reply: str) -> str | None:
    m = re.search(r"```(?:python)?\s*(.*?)```", reply, re.S)
    code = m.group(1) if m else reply
    return code if "def tool(" in code else None


def run_generated_tool(code: str, y: np.ndarray, args: dict,
                       timeout: float = 15.0):
    """Screen, then execute in an isolated subprocess via temp files."""
    if BANNED.search(code):
        return None, "banned construct in generated code"
    with tempfile.TemporaryDirectory() as td:
        np.save(os.path.join(td, "y.npy"), y)
        with open(os.path.join(td, "gen.py"), "w") as fh:
            fh.write("import numpy as np\nimport json, sys\n"
                     + code +
                     f"\ny = np.load(r'{td}/y.npy')\n"
                     f"args = json.loads('{json.dumps(args)}')\n"
                     "out = tool(y, *args.values())\n"
                     f"np.save(r'{td}/out.npy', np.asarray(out, "
                     "dtype=np.float64))\n")
        try:
            r = subprocess.run([sys.executable, "-I",
                                os.path.join(td, "gen.py")],
                               capture_output=True, timeout=timeout,
                               env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return None, "timeout"
        if r.returncode != 0:
            return None, (r.stderr.decode()[-200:] or "nonzero exit")
        try:
            out = np.load(os.path.join(td, "out.npy"))
        except Exception as e:
            return None, f"no output: {e}"
    return out, None


def score_d(task: dict, reply: str, y: np.ndarray):
    code = _extract_code(reply or "")
    if code is None:
        return False, "no function found"
    out, err = run_generated_tool(code, y, task["args"])
    if out is None:
        return False, err
    ref = np.asarray(task["ref"](y, task["args"]), dtype=np.float64)
    out = np.asarray(out, dtype=np.float64)
    if ref.size == 1:                       # scalar-result algorithms
        rel = abs(float(out.ravel()[0]) - float(ref.ravel()[0])) / \
            max(abs(float(ref.ravel()[0])), 1e-12)
        return rel <= task["tol"], f"rel err {rel:.4f}"
    if out.shape != ref.shape:
        return False, f"shape {out.shape} != {ref.shape}"
    denom = max(float(np.std(ref)), 1e-12)
    nrmse = float(np.sqrt(np.nanmean((out - ref) ** 2)) / denom)
    return nrmse <= task["tol"], f"NRMSE {nrmse:.4f}"


# --------------------------------------------------------------------------
# Family E - algorithm selection scenarios
# --------------------------------------------------------------------------
REGISTRY = ("detect_anomalies, channel_stats, zero_baseline, "
            "estimate_saturation, reconstruct_rlc, lowpass_filter, "
            "moving_average")

E_TASKS = [
    dict(
        name="de-censor 6.6kA",
        prompt=(
            "Scope Studio tool registry: " + REGISTRY + ".\n"
            "Situation: a capacitor-bank discharge. The busbar current "
            "monitor saturates softly above 6000 A (readings above that "
            "are compressed but still vary). The Pearson current "
            "transformer is faithful only for the first ~5 ms before its "
            "core saturates. The true peak is somewhere above both "
            "sensors' valid ranges. The user wants the full waveform "
            "reconstructed through the censored region.\n"
            "Which ONE tool from the registry is correct, and which "
            "parameter carries the busbar's 6000 A limit? Reply with "
            'ONLY JSON: {"run": "<tool>", "sat_level": <number>}.'),
        want={"run": "reconstruct_rlc", "sat_level": 6000},
    ),
    dict(
        name="offset removal",
        prompt=(
            "Scope Studio tool registry: " + REGISTRY + ".\n"
            "Situation: both channels show a constant nonzero level "
            "before the trigger; the user wants both signals to start "
            "at exactly 0 A. Which ONE tool is correct? Reply with ONLY "
            'JSON: {"run": "<tool>"}.'),
        want={"run": "zero_baseline"},
    ),
    dict(
        name="hard clip estimate",
        prompt=(
            "Scope Studio tool registry: " + REGISTRY + ".\n"
            "Situation: a sensor hard-clips flat at its range limit for "
            "30 ms; the user only needs a quick estimate of the true "
            "peak from the rise and droop slopes, not a full waveform. "
            'Which ONE tool? Reply with ONLY JSON: {"run": "<tool>"}.'),
        want={"run": "estimate_saturation"},
    ),
]


# --------------------------------------------------------------------------
# Family G - multi-step reasoning: no relation is handed over whole; the
# model must chain 2-3 elementary steps to a numeric answer (5% tol).
# --------------------------------------------------------------------------
G_TASKS = [
    dict(name="energy at peak",
         prompt=("A heavily overdamped capacitor-bank discharge: bank "
                 "voltage V0 = 450 V, total loop resistance R = 68 mOhm, "
                 "coil inductance L = 160 uH. Step 1: the peak current "
                 "is approximately V0/R. Step 2: the magnetic energy "
                 "stored in the coil at that current is E = 0.5*L*I^2. "
                 "Compute E in joules. Reply with only the number."),
         truth=0.5 * 160e-6 * (450 / 0.068) ** 2),          # ~3504 J
    dict(name="invert soft compression",
         prompt=("A current monitor compresses readings above 6000 A: "
                 "for true currents I > 6000, the reading is "
                 "R = 6000 + 0.25*(I - 6000). The monitor reads 6150 A. "
                 "What is the true current in amps? Reply with only the "
                 "number."),
         truth=6600.0),
    dict(name="Nyquist from record",
         prompt=("An oscilloscope records 1,250,000 samples spanning "
                 "exactly 200 ms. Step 1: compute the sample interval. "
                 "Step 2: the Nyquist frequency is 1/(2*dt). Give the "
                 "Nyquist frequency in kHz. Reply with only the "
                 "number."),
         truth=3125.0),
]


# --------------------------------------------------------------------------
# Family H - algorithm creation (harder than D): fully specified
# scientific algorithms the model must implement correctly.
# --------------------------------------------------------------------------
def _ref_despike(y, w, k):
    kern = np.ones(int(w)) / int(w)
    base = np.convolve(y, kern, mode="same")
    r = y - base
    mad = np.median(np.abs(r - np.median(r)))
    sig = 1.4826 * mad
    out = y.copy()
    bad = np.abs(r) > k * sig
    out[bad] = base[bad]
    return out


def _ref_trigger(y, dt):
    base = float(np.mean(y[: max(1, len(y) // 20)]))
    pk = float(np.max(y))
    thr = base + 0.5 * (pk - base)
    idx = int(np.flatnonzero(y >= thr)[0])
    return idx * dt


H_TASKS = [
    dict(
        name="MAD despike",
        prompt=(
            "Write a Python function using ONLY numpy (imported as np):\n\n"
            "def tool(y, w, k):\n"
            "    # 1) base = centered moving average of y, window w "
            "(np.convolve, mode='same')\n"
            "    # 2) r = y - base\n"
            "    # 3) sigma = 1.4826 * median(|r - median(r)|)   (MAD)\n"
            "    # 4) return a copy of y where samples with |r| > k*sigma "
            "are replaced by base at those samples\n\n"
            "Reply with ONLY the complete function in a python code "
            "block."),
        args={"w": 31, "k": 5.0},
        ref=lambda y, a: _ref_despike(y, a["w"], a["k"]),
        tol=0.02,
    ),
    dict(
        name="trigger time",
        prompt=(
            "Write a Python function using ONLY numpy (imported as np):\n\n"
            "def tool(y, dt):\n"
            "    # baseline = mean of the first 5% of samples\n"
            "    # threshold = baseline + 0.5*(max(y) - baseline)\n"
            "    # return the time (index*dt) of the FIRST sample where "
            "y >= threshold\n\n"
            "Reply with ONLY the complete function in a python code "
            "block."),
        args={"dt": 1.6e-7},
        ref=lambda y, a: _ref_trigger(y, a["dt"]),
        tol=0.01,
    ),
]


# --------------------------------------------------------------------------
# Family F - guided physics: the prompt supplies the governing relation;
# the model must apply it to the rig's numbers (discovery with guidance,
# not memorization). Numeric scoring, 5% tolerance.
# --------------------------------------------------------------------------
F_TASKS = [
    dict(name="rise time L/R",
         prompt=("A coil has L = 160 uH and the discharge loop has total "
                 "resistance R = 68 mOhm. The current rise time constant "
                 "of an overdamped RLC discharge is tau = L/R. Compute "
                 "tau in milliseconds. Reply with only the number."),
         truth=160e-6 / 0.068 * 1e3),                       # 2.353 ms
    dict(name="series-parallel C",
         prompt=("A capacitor bank has 4 modules in parallel; each "
                 "module is two 1.12 F capacitors in SERIES. Series "
                 "capacitance is C1*C2/(C1+C2); parallel capacitances "
                 "add. Compute the total bank capacitance in farads. "
                 "Reply with only the number."),
         truth=4 * (1.12 / 2)),                             # 2.24 F
    dict(name="implied C from droop",
         prompt=("A capacitor-bank discharge shows a fitted droop time "
                 "constant tau_droop = 387 ms through a total resistance "
                 "R = 68 mOhm. Using tau_droop = R*C, compute the "
                 "implied capacitance C in farads. Reply with only the "
                 "number."),
         truth=0.387 / 0.068),                              # 5.69 F
]


def score_f(task: dict, reply: str):
    m = re.search(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", (reply or "").replace(
        ",", ""))
    if not m:
        return False, "no number"
    val = float(m.group())
    ok = abs(val - task["truth"]) / max(abs(task["truth"]), 1e-12) <= 0.05
    return ok, f"{val:g} (truth {task['truth']:.4g})"


def score_e(task: dict, reply: str):
    m = re.search(r"\{.*?\}", reply or "", re.S)
    if not m:
        return False, "no JSON"
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return False, "bad JSON"
    for k, v in task["want"].items():
        got = obj.get(k)
        if isinstance(v, (int, float)):
            try:
                if abs(float(got) - v) > 1e-6:
                    return False, json.dumps(obj)
            except (TypeError, ValueError):
                return False, json.dumps(obj)
        elif str(got) != str(v):
            return False, json.dumps(obj)
    return True, json.dumps(obj)


# --------------------------------------------------------------------------
def ask(backend: str, model: str, prompt: str) -> str:
    from ai_assistant import ask_model
    return ask_model(prompt, model=model, backend=backend,
                     system_prompt="You are a precise engineering "
                     "assistant. Follow the output format exactly.",
                     max_tokens=900)


class Mock:
    def __init__(self, mode): self.mode = mode
    def __call__(self, task, fam):
        if self.mode == "perfect":
            if fam == "D" and "moving" in task["name"]:
                return ("```python\ndef tool(y, w):\n    k = np.ones(int(w))"
                        "/int(w)\n    return np.convolve(y, k, mode='same')\n```")
            if fam == "D":
                return ("```python\ndef tool(y, dt, fc):\n    a = 2*np.pi*fc*dt/"
                        "(2*np.pi*fc*dt+1)\n    out = np.empty_like(y)\n    "
                        "out[0] = y[0]\n    for i in range(1, len(y)):\n        "
                        "out[i] = out[i-1] + a*(y[i]-out[i-1])\n    return out\n```")
            if fam in ("F", "G"):
                return f"{task['truth']:.6g}"
            if fam == "H" and "despike" in task["name"]:
                return ("```python\ndef tool(y, w, k):\n"
                        "    kern = np.ones(int(w))/int(w)\n"
                        "    base = np.convolve(y, kern, mode='same')\n"
                        "    r = y - base\n"
                        "    sig = 1.4826*np.median(np.abs(r-np.median(r)))\n"
                        "    out = y.copy()\n"
                        "    bad = np.abs(r) > k*sig\n"
                        "    out[bad] = base[bad]\n    return out\n```")
            if fam == "H":
                return ("```python\ndef tool(y, dt):\n"
                        "    base = np.mean(y[:max(1, len(y)//20)])\n"
                        "    thr = base + 0.5*(np.max(y)-base)\n"
                        "    return float(np.flatnonzero(y >= thr)[0])*dt\n```")
            return json.dumps(task["want"])
        if fam == "D":
            return "```python\ndef tool(y, *a):\n    import os\n    return y\n```"
        if fam == "F":
            return "42"
        return '{"run": "detect_anomalies"}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mlx",
                    choices=["mlx", "ollama", "llama.cpp"])
    ap.add_argument("--models", default="",
                    help="comma list: MLX folders / ollama tags")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "backtests", "tool_creation_benchmark.txt"))
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    y = (np.sin(2 * np.pi * 3 * np.linspace(0, 1, 8000))
         * 3000 + rng.normal(0, 40, 8000))
    # H needs structure: a pulse with injected spikes for despike/trigger
    y_h = np.where(np.arange(8000) > 1500, 5000.0, 0.0) + \
        rng.normal(0, 20, 8000)
    y_h[np.array([2200, 3300, 4400, 5500])] += 2500.0   # known spikes

    models = ([Mock("perfect"), Mock("naive")] if args.mock else
              [m.strip() for m in
               os.path.expanduser(args.models).split(",") if m.strip()])
    if not models:
        print("Give --models (or --mock). For MLX pass model FOLDERS.")
        return 2

    lines, summary = [], {}
    for mdl in models:
        name = mdl.mode if isinstance(mdl, Mock) else os.path.basename(
            str(mdl).rstrip("/"))
        n_ok = 0
        for fam, tasks, scorer in (("D", D_TASKS,
                                    lambda t, r: score_d(t, r, y)),
                                   ("E", E_TASKS,
                                    lambda t, r: score_e(t, r)),
                                   ("F", F_TASKS,
                                    lambda t, r: score_f(t, r)),
                                   ("G", G_TASKS,
                                    lambda t, r: score_f(t, r)),
                                   ("H", H_TASKS,
                                    lambda t, r: score_d(t, r, y_h))):
            for t in tasks:
                t0 = time.perf_counter()
                reply = (mdl(t, fam) if isinstance(mdl, Mock)
                         else ask(args.backend, mdl, t["prompt"]))
                el = time.perf_counter() - t0
                # backend/load errors must surface as errors, not be
                # scored as answers (e.g. '-3' parsed out of an error
                # string mentioning 'Nemotron-3')
                err_markers = ("AI backend error", "is not installed",
                               "not reachable", "Traceback",
                               "No safetensors", "Error:")
                if any(m in (reply or "") for m in err_markers) \
                        or not (reply or "").strip():
                    ok, detail = False, ("BACKEND ERROR: "
                                         + (reply or "(empty)")[:120])
                else:
                    ok, detail = scorer(t, reply)
                # surface backend/load errors instead of a bare "no JSON"
                if not ok and detail in ("no JSON", "no number",
                                         "no function found") and reply:
                    detail += f" | reply: {reply[:90]!r}"
                n_ok += ok
                lines.append(f"[{'PASS' if ok else 'FAIL'}] {name:28s} "
                             f"{fam} {t['name']:22s} {detail} "
                             f"({el:.1f}s)")
        summary[name] = n_ok
    total = (len(D_TASKS) + len(E_TASKS) + len(F_TASKS)
             + len(G_TASKS) + len(H_TASKS))
    hdr = ["Tool/algorithm-creation, selection, physics & reasoning "
           f"benchmark (backend={args.backend})", ""]
    hdr += [f"  {k:30s} {v}/{total}" for k, v in summary.items()] + [""]
    text = "\n".join(hdr + lines)
    print(text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
