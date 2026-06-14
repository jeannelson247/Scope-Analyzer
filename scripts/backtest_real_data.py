#!/usr/bin/env python3
"""
backtest_real_data.py - Run the Scope Studio analysis pipeline on real
oscilloscope shots (Tektronix DPO2024B exports in the "Data Scope" folder).

Stages per file:
  1. csv_loader.load_csv      - preamble parsing, units, big-file handling
  2. baseline correction      - pre-trigger (t < t0) mean subtracted from
                                current-like channels (signal_tools.baseline)
  3. pulse statistics         - peak, plateau mean (samples > plateau_frac
                                of peak), pulse duration, charge integral
  4. detect_anomalies.detect  - spikes / clipping / drift / crest / balance
  5. benchmark check          - optional expected plateau current with
                                tolerance (e.g. 6.6 kA shot)

Usage:
  python3 scripts/backtest_real_data.py <csv-or-folder> [more ...]
          [--expect-amps 6600] [--tolerance 0.05] [--out report.txt]

With no arguments it looks for the two standard Data Scope folders next to
the repo (see DEFAULT_SOURCES below).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from csv_loader import load_csv                      # noqa: E402
from detect_anomalies import detect                  # noqa: E402
from signal_tools import baseline                    # noqa: E402

DEFAULT_SOURCES = [
    os.path.expanduser(
        "~/Documents/Data Scope/2026-04-20 4 Modules full amperage @ 100% 6.6kA"),
    os.path.expanduser(
        "~/Documents/Data Scope/2026-04-09 4 Modules in parallel 100% "
        "and step current waaveforms"),
]

PLATEAU_FRAC = 0.90       # plateau = samples above this fraction of peak
PULSE_FRAC = 0.50         # pulse extent = first/last crossing of this level


def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy 2.x keeps trapezoidal integration as `trapezoid`; older
    environments may only have `trapz`. Use the modern spelling first so
    the real-data benchmark works across the launch matrix."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # pragma: no cover - legacy NumPy fallback


def is_current(label: str, unit: str) -> bool:
    return unit.upper().startswith("A") or "current" in label.lower()


def pulse_stats(t: np.ndarray, y: np.ndarray) -> dict:
    """Peak / plateau / duration / charge for a single (baseline-corrected)
    current channel. Sign-aware: works for negative pulses too."""
    pk_idx = int(np.nanargmax(np.abs(y)))
    pk = float(y[pk_idx])
    sgn = 1.0 if pk >= 0 else -1.0
    ys = sgn * y
    pk_abs = ys[pk_idx]
    plateau_mask = ys >= PLATEAU_FRAC * pk_abs
    plateau_mean = sgn * float(np.nanmean(ys[plateau_mask]))
    above = np.flatnonzero(ys >= PULSE_FRAC * pk_abs)
    t_start, t_end = float(t[above[0]]), float(t[above[-1]])
    charge = integrate_trapezoid(y, t)
    return {
        "peak_A": pk,
        "plateau_A": plateau_mean,
        "n_plateau": int(plateau_mask.sum()),
        "t_start_s": t_start,
        "t_end_s": t_end,
        "duration_s": t_end - t_start,
        "charge_C": charge,
    }


def analyze_file(path: str, expect_amps: float | None,
                 tolerance: float, lines: list[str]) -> bool:
    """Returns True when the benchmark check (if any) passes."""
    lines.append("=" * 72)
    lines.append(f"FILE: {path}")
    d = load_csv(path)
    t = d.df.iloc[:, 0].to_numpy(np.float64)
    model = (d.meta.get("Model") or ["?"])[0]
    dt = float(np.median(np.diff(t)))
    lines.append(f"  scope: {model} | rows: {d.n_rows:,} | dt: {dt:.3g} s "
                 f"| span: {t[0]:.4g} .. {t[-1]:.4g} s")
    lines.append(f"  columns: {d.columns} | units: {d.units}")

    # Channels: skip 'Peak Detect' envelope columns for stats (they are
    # min/max interleaved and would distort means), but keep them for the
    # anomaly scan's clipping check? -> keep stats channels only.
    chan_labels = [c for c in d.columns[1:] if "peak detect" not in c.lower()]
    ok = True
    channels_for_scan: dict[str, np.ndarray] = {}
    pre_trigger_end = min(0.0, t[0] + 0.25 * (t[-1] - t[0]))

    for c in chan_labels:
        y = d.df[c].to_numpy(np.float64)
        unit = d.units.get(c, "V")
        label = f"{c} ({unit})"
        if is_current(c, unit):
            y0 = baseline(y, t, end=pre_trigger_end)
            off = float(np.nanmean(y[t <= pre_trigger_end])) if \
                np.any(t <= pre_trigger_end) else 0.0
            st = pulse_stats(t, y0)
            lines.append(
                f"  {label}: baseline offset {off:+.4g} {unit} removed | "
                f"peak {st['peak_A']:.6g} {unit} | "
                f"plateau {st['plateau_A']:.6g} {unit} "
                f"({st['n_plateau']:,} samples) | "
                f"pulse {st['duration_s']*1e3:.3g} ms "
                f"({st['t_start_s']*1e3:.3g} .. {st['t_end_s']*1e3:.3g} ms) | "
                f"charge {st['charge_C']:.4g} C")
            channels_for_scan[label] = y0
            if expect_amps is not None:
                err = abs(abs(st["plateau_A"]) - expect_amps) / expect_amps
                verdict = "PASS" if err <= tolerance else "FAIL"
                if verdict == "FAIL":
                    ok = False
                lines.append(
                    f"    BENCHMARK {verdict}: plateau "
                    f"{abs(st['plateau_A']):.6g} {unit} vs expected "
                    f"{expect_amps:g} (error {100*err:.2f}%, "
                    f"tolerance {100*tolerance:g}%)")
        else:
            lines.append(
                f"  {label}: min {np.nanmin(y):.4g} max {np.nanmax(y):.4g} "
                f"mean {np.nanmean(y):.4g} {unit}")
            channels_for_scan[label] = y

    rep = detect(t * 1e3, channels_for_scan, x_unit="ms")
    lines.append("  " + rep.text().replace("\n", "\n  "))
    return ok


def collect(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.CSV"))) +
                       sorted(glob.glob(os.path.join(p, "*.csv"))))
        elif os.path.isfile(p):
            out.append(p)
    # de-dup, keep order
    seen, files = set(), []
    for f in out:
        if f not in seen:
            seen.add(f)
            files.append(f)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*", default=None)
    ap.add_argument("--expect-amps", type=float, default=None,
                    help="expected plateau current for benchmark check")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sources = args.sources or [s for s in DEFAULT_SOURCES if os.path.isdir(s)]
    files = collect(sources)
    if not files:
        print("No CSV files found in:", sources)
        return 2

    lines: list[str] = [f"Scope Studio real-data backtest "
                        f"({len(files)} file(s))"]
    n_pass = n_fail = 0
    for f in files:
        try:
            if analyze_file(f, args.expect_amps, args.tolerance, lines):
                n_pass += 1
            else:
                n_fail += 1
        except Exception as exc:  # keep going; report the failure
            n_fail += 1
            lines.append(f"  ERROR processing {f}: {exc!r}")
    lines.append("=" * 72)
    lines.append(f"SUMMARY: {n_pass} pass / {n_fail} fail "
                 f"(benchmark={'%g A' % args.expect_amps if args.expect_amps else 'none'})")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
