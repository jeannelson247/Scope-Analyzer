#!/usr/bin/env python3
"""
benchmark_data_analysis.py - Compare lightweight local models on scope-data
analysis tasks, scored against NumPy ground truth from a REAL shot CSV.

Three task families, matching how Scope Studio actually uses an LLM:

  A. stats-reading   - the model gets the same deterministic stats text the
                       side chat shows, and must answer questions whose
                       answers are in that text (peak, average, spike count,
                       pulse duration). Tests faithful extraction.
  B. tool-routing    - the model must emit a JSON action block that routes
                       to the right deterministic tool with the right
                       parameters. Tests the "LLM never computes" design.
  C. raw-arithmetic  - the model gets a 48-sample downsampled array and must
                       compute average / max / #values above a threshold
                       itself. Small models are EXPECTED to be unreliable
                       here; this family exists to quantify exactly why the
                       deterministic-tools rule is correct.

Each numeric answer is scored with a relative tolerance (default 2%).
The user remains the sanity check: the report prints model answer vs truth
side by side for every question.

Usage (on the Mac, with Ollama running):
  python3 scripts/benchmark_data_analysis.py \
      --csv "~/Documents/Data Scope/2026-04-20 4 Modules full amperage @ 100% 6.6kA/T0000.CSV"
  python3 scripts/benchmark_data_analysis.py --models llama3.2:1b,qwen2.5:0.5b
  python3 scripts/benchmark_data_analysis.py --mock     # no models needed;
                                                        # validates scoring

Models are auto-detected from `ollama list`; only installed candidates run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from csv_loader import load_csv                  # noqa: E402
from detect_anomalies import detect              # noqa: E402
from signal_tools import baseline                # noqa: E402

DEFAULT_CSV = os.path.expanduser(
    "~/Documents/Data Scope/2026-04-20 4 Modules full amperage @ 100% 6.6kA/"
    "T0000.CSV")

# Lightweight candidates to compare (ollama tags). Only installed ones run.
CANDIDATE_MODELS = [
    "functiongemma",       # 270M action router (current catalog tier 1)
    "llama3.2:1b",         # current catalog tier 2
    "llama3.2",            # 3B, current catalog default
    "qwen2.5:0.5b",        # alternative tiny
    "qwen2.5:1.5b",        # alternative light
    "gemma3:1b",           # alternative light
    "smollm2:1.7b",        # alternative light
    "qwen3.5:9b",          # heavyweight reference point (~6.6 GB Q4_K_M)
    "qwen3:14b",           # 14B step-up (~9.3 GB Q4) - close other apps
    "qwen3.5:4b-mlx",      # MLX runner, 4.0 GB - Apple Silicon fast path
    "qwen3.5:9b-mlx",      # MLX runner, 8.9 GB - heavy tier, fast decode
]

REL_TOL = 0.02  # 2% relative tolerance on numeric answers


# --------------------------------------------------------------------------
# Ground truth from the real CSV
# --------------------------------------------------------------------------
def ground_truth(csv_path: str) -> dict:
    d = load_csv(csv_path)
    t = d.df.iloc[:, 0].to_numpy(np.float64)
    cur_col = next((c for c in d.columns[1:]
                    if d.units.get(c, "").upper().startswith("A")
                    and "peak detect" not in c.lower()), None)
    if cur_col is None:
        raise SystemExit("No current (A) channel found in CSV.")
    y_raw = d.df[cur_col].to_numpy(np.float64)
    pre_end = min(0.0, t[0] + 0.25 * (t[-1] - t[0]))
    y = baseline(y_raw, t, end=pre_end)

    pk = float(np.max(np.abs(y)))
    plateau = float(np.mean(np.abs(y)[np.abs(y) >= 0.9 * pk]))
    above = np.flatnonzero(np.abs(y) >= 0.5 * pk)
    dur_ms = float((t[above[-1]] - t[above[0]]) * 1e3)
    mean_all = float(np.mean(y))
    rep = detect(t * 1e3, {f"{cur_col} (A)": y}, x_unit="ms")
    n_spikes = rep.findings[0].n_spike_events

    # 48-sample downsample for the raw-arithmetic family (decimate by mean)
    idx = np.linspace(0, len(y), 49).astype(int)
    small = np.array([float(np.mean(y[a:b])) for a, b in
                      zip(idx[:-1], idx[1:])])
    small = np.round(small, 1)

    stats_text = (
        f"Channel {cur_col} (A), baseline-corrected, "
        f"{len(y):,} samples, {t[0]*1e3:.2f}..{t[-1]*1e3:.2f} ms\n"
        f"  peak current: {pk:.1f} A\n"
        f"  plateau mean (>=90% of peak): {plateau:.1f} A\n"
        f"  full-window average: {mean_all:.1f} A\n"
        f"  pulse duration (50% level): {dur_ms:.2f} ms\n"
        f"  spike events above 6 sigma: {n_spikes}\n")

    return {
        "csv": csv_path, "channel": cur_col, "stats_text": stats_text,
        "peak": pk, "plateau": plateau, "mean_all": mean_all,
        "dur_ms": dur_ms, "n_spikes": n_spikes, "small": small,
    }


def build_tasks(gt: dict) -> list[dict]:
    small = gt["small"]
    thresh = round(float(np.percentile(small, 80)), 1)
    return [
        # A. stats-reading -------------------------------------------------
        dict(family="A stats-reading", q=(
            gt["stats_text"] + "\nQuestion: what is the peak current in "
            "amps? Reply with only the number."),
            truth=gt["peak"], kind="num"),
        dict(family="A stats-reading", q=(
            gt["stats_text"] + "\nQuestion: what is the pulse duration in "
            "ms? Reply with only the number."),
            truth=gt["dur_ms"], kind="num"),
        dict(family="A stats-reading", q=(
            gt["stats_text"] + "\nQuestion: how many spike events above "
            "6 sigma were detected? Reply with only the integer."),
            truth=float(gt["n_spikes"]), kind="int"),
        dict(family="A stats-reading", q=(
            gt["stats_text"] + "\nQuestion: what is the plateau mean "
            "current in amps? Reply with only the number."),
            truth=gt["plateau"], kind="num"),
        # B. tool-routing ----------------------------------------------------
        dict(family="B tool-routing", q=(
            "You control Scope Studio. The user says: 'scan the current "
            "channel for spikes with a 5 sigma threshold'. Reply with ONLY "
            "a JSON object of the form "
            '{"run": "detect_anomalies", "threshold_sigma": <number>}.'),
            truth={"run": "detect_anomalies", "threshold_sigma": 5},
            kind="json"),
        dict(family="B tool-routing", q=(
            "You control Scope Studio. The user says: 'average the visible "
            "current channel'. Reply with ONLY a JSON object of the form "
            '{"run": "channel_stats", "stat": "mean"}.'),
            truth={"run": "channel_stats", "stat": "mean"}, kind="json"),
        # C. raw-arithmetic ------------------------------------------------
        dict(family="C raw-arithmetic", q=(
            "Data (A): " + ", ".join(f"{v:g}" for v in small) +
            "\nQuestion: what is the average of these values? Reply with "
            "only the number."),
            truth=float(np.mean(small)), kind="num"),
        dict(family="C raw-arithmetic", q=(
            "Data (A): " + ", ".join(f"{v:g}" for v in small) +
            "\nQuestion: what is the maximum value? Reply with only the "
            "number."),
            truth=float(np.max(small)), kind="num"),
        dict(family="C raw-arithmetic", q=(
            "Data (A): " + ", ".join(f"{v:g}" for v in small) +
            f"\nQuestion: how many values are greater than {thresh}? "
            "Reply with only the integer."),
            truth=float(np.sum(small > thresh)), kind="int"),
    ]


# --------------------------------------------------------------------------
# Model runners
# --------------------------------------------------------------------------
def installed_ollama_models() -> list[str]:
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return []
    names = [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip()]
    hits = []
    for cand in CANDIDATE_MODELS:
        base = cand.split(":")[0]
        if any(n == cand or n.startswith(cand + ":") or
               n.split(":")[0] == base and ":" not in cand
               for n in names):
            hits.append(cand)
    return hits


THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def ask_ollama(model: str, prompt: str, timeout: float = 240.0) -> str:
    """Query Ollama. Thinking models (qwen3/qwen3.5, ...) reason before
    answering; with a small num_predict they burn the whole budget on
    reasoning and return nothing. So: ask Ollama to disable thinking
    ("think": false), give a generous budget as backstop, and strip any
    <think> blocks that leak into the response before scoring."""
    import urllib.error
    import urllib.request

    def _call(payload: dict) -> str:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response", "")

    base = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "num_predict": 1024}}
    try:
        reply = _call({**base, "think": False})
    except urllib.error.HTTPError:
        # older Ollama or model that rejects the think flag - plain call
        reply = _call(base)
    return THINK_RE.sub("", reply).strip()


class MockModel:
    """Validates the scoring pipeline without any LLM. 'perfect' answers
    from ground truth; 'naive' guesses plausibly-wrong values - the report
    must score the first ~100% and the second low."""

    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, task: dict) -> str:
        if task["kind"] == "json":
            return (json.dumps(task["truth"]) if self.mode == "perfect"
                    else '{"run": "set_plot_style"}')
        v = task["truth"]
        if self.mode == "perfect":
            return f"{v:.2f}"
        return f"{v * 1.37 + 11:.2f}"  # wrong by construction


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?", re.I)


def first_number(text: str) -> float | None:
    m = NUM_RE.search(text.replace(",", ""))
    return float(m.group()) if m else None


def score(task: dict, reply: str) -> tuple[bool, str]:
    if task["kind"] == "json":
        m = re.search(r"\{.*\}", reply, re.S)
        if not m:
            return False, "no JSON found"
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return False, "invalid JSON"
        want = task["truth"]
        ok = all(str(obj.get(k)) == str(v) for k, v in want.items())
        return ok, json.dumps(obj)
    val = first_number(reply)
    if val is None:
        return False, "no number found"
    truth = task["truth"]
    if task["kind"] == "int":
        return val == truth, f"{val:g}"
    denom = max(abs(truth), 1e-9)
    return abs(val - truth) / denom <= REL_TOL, f"{val:g}"


def run_benchmark(models: list, tasks: list[dict], asker) -> list[dict]:
    rows = []
    for model in models:
        name = model if isinstance(model, str) else f"mock:{model.mode}"
        for i, task in enumerate(tasks):
            t0 = time.perf_counter()
            try:
                reply = (model(task) if callable(model)
                         else asker(model, task["q"]))
                err = None
            except Exception as exc:
                reply, err = "", repr(exc)
            dt = time.perf_counter() - t0
            ok, parsed = (False, err) if err else score(task, reply)
            rows.append(dict(model=name, task=i, family=task["family"],
                             ok=ok, answer=parsed,
                             truth=(task["truth"] if task["kind"] != "json"
                                    else json.dumps(task["truth"])),
                             seconds=round(dt, 2)))
    return rows


def render(rows: list[dict], gt: dict) -> str:
    lines = ["Scope Studio - lightweight model data-analysis benchmark",
             f"source: {gt['csv']}",
             f"ground truth: peak {gt['peak']:.1f} A | plateau "
             f"{gt['plateau']:.1f} A | duration {gt['dur_ms']:.2f} ms | "
             f"spikes {gt['n_spikes']}", ""]
    models = sorted({r["model"] for r in rows})
    fams = sorted({r["family"] for r in rows})
    hdr = f"{'model':24s} " + " ".join(f"{f.split()[0]:>8s}" for f in fams) \
        + f" {'total':>7s} {'avg s':>6s}"
    lines += [hdr, "-" * len(hdr)]
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        cells = []
        for f in fams:
            sub = [r for r in mine if r["family"] == f]
            cells.append(f"{sum(r['ok'] for r in sub)}/{len(sub)}")
        tot = f"{sum(r['ok'] for r in mine)}/{len(mine)}"
        avg = np.mean([r["seconds"] for r in mine])
        lines.append(f"{m:24s} " + " ".join(f"{c:>8s}" for c in cells) +
                     f" {tot:>7s} {avg:>6.2f}")
    lines.append("")
    lines.append("Per-question detail (model answer vs truth - sanity-check "
                 "these by eye):")
    for r in rows:
        mark = "PASS" if r["ok"] else "FAIL"
        lines.append(f"  [{mark}] {r['model']:22s} {r['family']:17s} "
                     f"answer={r['answer']} truth={r['truth']} "
                     f"({r['seconds']}s)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--models", default=None,
                    help="comma list of ollama tags (default: auto-detect)")
    ap.add_argument("--mock", action="store_true",
                    help="run mock perfect/naive models (no LLM needed)")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "backtests", "model_data_benchmark.txt"))
    args = ap.parse_args()

    csv_path = os.path.expanduser(args.csv)
    if not os.path.isfile(csv_path):
        print(f"CSV not found: {csv_path}")
        return 2
    gt = ground_truth(csv_path)
    tasks = build_tasks(gt)

    if args.mock:
        models: list = [MockModel("perfect"), MockModel("naive")]
        asker = None
    else:
        names = (args.models.split(",") if args.models
                 else installed_ollama_models())
        if not names:
            print("No candidate models installed in Ollama. Install some, "
                  "e.g.:\n  ollama pull llama3.2:1b\n  ollama pull "
                  "qwen2.5:0.5b\nor run with --mock to validate scoring.")
            return 2
        models, asker = names, ask_ollama

    rows = run_benchmark(models, tasks, asker)
    text = render(rows, gt)
    print(text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(text + "\n")
    print(f"\nreport written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
