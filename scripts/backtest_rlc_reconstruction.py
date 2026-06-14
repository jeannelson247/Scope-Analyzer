#!/usr/bin/env python3
"""Real-data benchmark for the censored RLC reconstruction path.

This benchmark is designed for the 6.6 kA shot workflow:

- target: a BBCM/busbar channel converted to current
- reference: Pearson current trusted before a chosen cutoff
- gap: the interval we want to reconstruct as an overlay/model estimate
- post: clean later data that the model should remain faithful to

It reports evidence and failure modes. It does not claim the RLC model proves
the waveform; it only checks whether the deterministic overlay is consistent
with the chosen trusted windows and censoring assumptions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from csv_loader import load_csv  # noqa: E402
from rlc_reconstruct import fit_rlc, rlc_model  # noqa: E402
from signal_tools import baseline, evaluate_formula  # noqa: E402

DEFAULT_SOURCE = (
    "~/Documents/Data Scope/2026-04-20 4 Modules full amperage @ 100% 6.6kA"
)


def first_csv(source: str) -> str:
    path = Path(os.path.expanduser(source))
    if path.is_file():
        return str(path)
    for name in ("T0000.CSV", "T0000.csv"):
        candidate = path / name
        if candidate.is_file():
            return str(candidate)
    matches = sorted(path.glob("*.CSV")) + sorted(path.glob("*.csv"))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"No CSV found in {source}")


def load_preset(name: str) -> dict:
    with open(ROOT / "presets.json") as fh:
        presets = json.load(fh)
    if name not in presets:
        raise KeyError(f"Preset not found: {name}")
    return presets[name]


def nrmse(model: np.ndarray, data: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    resid = model[mask] - data[mask]
    rmse = float(np.sqrt(np.nanmean(resid ** 2)))
    denom = max(float(np.nanmax(np.abs(data[mask]))), 1.0)
    return rmse, rmse / denom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=DEFAULT_SOURCE)
    ap.add_argument("--target-column", default="CH1")
    ap.add_argument("--target-preset", default="BBCM v2")
    ap.add_argument("--ref-column", default="CH2")
    ap.add_argument("--sat-level", type=float, default=6000.0)
    ap.add_argument("--fit-start-ms", type=float, default=0.0)
    ap.add_argument("--fit-end-ms", type=float, default=150.0)
    ap.add_argument("--ref-start-ms", type=float, default=0.0)
    ap.add_argument("--ref-end-ms", type=float, default=5.0)
    ap.add_argument("--gap-start-ms", type=float, default=5.0)
    ap.add_argument("--gap-end-ms", type=float, default=40.0)
    ap.add_argument("--post-start-ms", type=float, default=40.0)
    ap.add_argument("--post-end-ms", type=float, default=150.0)
    ap.add_argument("--out", default="backtests/rlc_reconstruction_6p6kA_report.txt")
    args = ap.parse_args()

    csv_path = first_csv(args.source)
    data = load_csv(csv_path)
    t_s = data.df[data.columns[0]].to_numpy(np.float64)
    t_ms = t_s * 1000.0

    preset = load_preset(args.target_preset)
    raw_target = data.df[args.target_column].to_numpy(np.float64)
    target = evaluate_formula(preset.get("formula", "x"), raw_target, t_s, t_ms)
    target = target * float(preset.get("gain", 1.0)) + float(preset.get("offset", 0.0))
    target = baseline(target, t_ms, end=0.0)

    ref = data.df[args.ref_column].to_numpy(np.float64)
    ref = baseline(ref, t_ms, end=0.0)

    rep = fit_rlc(
        t_ms,
        target,
        sat_level=args.sat_level,
        label=f"{args.target_column} {args.target_preset}",
        t_window=(args.fit_start_ms, args.fit_end_ms),
        y_ref=ref,
        ref_window=(args.ref_start_ms, args.ref_end_ms),
        ref_label=args.ref_column,
    )

    lines = [
        "Scope Studio RLC reconstruction benchmark",
        "=" * 72,
        f"FILE: {csv_path}",
        f"target: {args.target_column} via preset '{args.target_preset}'",
        f"reference: {args.ref_column}, trusted {args.ref_start_ms:g}..{args.ref_end_ms:g} ms",
        f"censoring lower bound: {args.sat_level:g} A",
        f"fit window: {args.fit_start_ms:g}..{args.fit_end_ms:g} ms",
        f"reconstruction focus: {args.gap_start_ms:g}..{args.gap_end_ms:g} ms",
        "",
        rep.text,
        "",
        "Fidelity checks:",
    ]
    ok = bool(rep.ok)
    if rep.ok:
        params = rep.params
        model = rlc_model(
            t_ms,
            params["A"],
            params["t0"],
            params["tau_r"],
            params["tau_d"],
        )
        pre = (t_ms >= args.ref_start_ms) & (t_ms <= args.ref_end_ms)
        gap = (t_ms >= args.gap_start_ms) & (t_ms <= args.gap_end_ms)
        post = (t_ms >= args.post_start_ms) & (t_ms <= args.post_end_ms)

        pre_rmse, pre_nrmse = nrmse(model, ref, pre)
        gap_rmse, gap_nrmse = nrmse(model, target, gap)
        post_rmse, post_nrmse = nrmse(model, target, post)
        lower = np.minimum(target[gap], args.sat_level)
        gap_violation_pct = 100.0 * float(np.mean(model[gap] < lower))
        clean_pct_peak = 100.0 * float(params["rms"] / max(abs(params["peak"]), 1.0))

        checks = {
            "pre_ref_nrmse": pre_nrmse,
            "gap_lower_bound_violation_pct": gap_violation_pct,
            "post_target_nrmse": post_nrmse,
            "clean_residual_pct_peak": clean_pct_peak,
        }
        ok = (
            pre_nrmse <= 0.05
            and post_nrmse <= 0.08
            and gap_violation_pct <= 0.5
            and clean_pct_peak <= 5.0
        )
        lines.extend([
            f"- before {args.ref_end_ms:g} ms vs Pearson reference: "
            f"RMSE {pre_rmse:.3g} A, NRMSE {100*pre_nrmse:.2f}%",
            f"- {args.gap_start_ms:g}..{args.gap_end_ms:g} ms focus interval "
            f"vs BBCM/censored lower-bound signal: RMSE {gap_rmse:.3g} A, "
            f"NRMSE {100*gap_nrmse:.2f}%",
            f"- lower-bound violations in focus interval: {gap_violation_pct:.3g}%",
            f"- after {args.post_start_ms:g} ms vs BBCM target: "
            f"RMSE {post_rmse:.3g} A, NRMSE {100*post_nrmse:.2f}%",
            f"- clean-fit residual: {clean_pct_peak:.2f}% of reconstructed peak",
            "",
            "Verdict: " + ("PASS" if ok else "FAIL"),
        ])
        lines.append("Thresholds: pre <=5%, post <=8%, gap lower-bound violations <=0.5%, clean residual <=5% of peak.")
        lines.append("Metrics JSON: " + json.dumps(checks, sort_keys=True))
    else:
        lines.append("Verdict: FAIL (fit did not complete)")

    text = "\n".join(lines)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
