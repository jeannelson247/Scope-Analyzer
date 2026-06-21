"""Generate advanced synthetic stress-test CSVs for Scope Analyzer Lite.

The regular ``tool_benchmarks`` pack is intentionally friendly and tutorial
oriented. This stress pack is harsher: larger traces, bad timestamps, NaNs,
hidden clipping, high dynamic range, drift, ringing, and multi-channel skew.

All files are synthetic demonstration data. They are not experimental truth and
they never replace or modify a user's source oscilloscope CSV.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_lite_toolbox_examples import (
    L_INTERNAL_H,
    R_DEFAULT_OHM,
    control_pulse,
    rlc_pulse,
    write_scope_csv,
)

DEFAULT_OUT = ROOT / "examples" / "tool_stress"


def _guide(tool: str, column: str, why: str, params: dict | None = None,
           expected: str = "") -> dict[str, object]:
    return {
        "tool": tool,
        "column": column,
        "why": why,
        "params": params or {},
        "expected": expected or "The tool should complete without modifying the source CSV.",
    }


def make_stress_examples(out_dir: Path = DEFAULT_OUT) -> list[dict[str, object]]:
    rng = np.random.default_rng(20260622)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    def add(file: str, title: str, tools: list[str], guide: dict[str, object]) -> None:
        manifest.append({
            "id": f"S{len(manifest) + 1:02d}",
            "file": file,
            "title": title,
            "tools": tools,
            "guide": guide,
        })

    # S01: large-ish trace with narrow spikes to test loading, decimation, stats.
    t = np.linspace(0.0, 0.250, 80_000)
    current = rlc_pulse(t, 0.010, 0.006, 0.180, 5200.0)
    current += 35.0 * rng.normal(size=t.size)
    for center, amp in [(0.037, 1800.0), (0.121, -1300.0), (0.204, 1100.0)]:
        idx = np.argmin(np.abs(t - center))
        current[idx:idx + 3] += amp
    write_scope_csv(out_dir / "stress_01_large_decimation_spikes.csv", {
        "TIME": t,
        "Current_A": current,
        "Control_V": control_pulse(t, 0.010, 0.180, 5.0),
    }, {"Current_A": "A", "Control_V": "V"},
       {"purpose": "large trace, decimation, sparse spikes"})
    add("stress_01_large_decimation_spikes.csv", "Large trace with sparse spikes",
        ["stats", "anomaly", "lowpass"],
        _guide("anomaly", "Current_A",
               "Stress anomaly detection and display decimation on a much larger trace.",
               {"threshold_sigma": 6},
               "Injected spikes should be reported while the CSV hash stays unchanged."))

    # S02: nonuniform time with duplicate/backward samples and a gap.
    t = np.linspace(0.0, 0.030, 5000)
    t[1200] = t[1199]
    t[2200] = t[2199] - 3e-5
    t[3600:] += 0.006
    sig = 0.8 * np.sin(2 * np.pi * 800 * t)
    sig[2600:2610] = np.nan
    write_scope_csv(out_dir / "stress_02_nonuniform_time_nan.csv", {
        "TIME": t,
        "Signal_V": sig,
    }, {"Signal_V": "V"}, {"purpose": "quality report stress case"})
    add("stress_02_nonuniform_time_nan.csv", "Nonuniform time + NaNs",
        ["quality"], _guide("quality", "Signal_V",
                            "Verify QC catches duplicate/backward time samples, gaps, and NaNs."))

    # S03: FFT with fixed spur plus chirp and broadband noise.
    fs = 500_000.0
    t = np.arange(0.0, 0.040, 1 / fs)
    chirp_phase = 2 * np.pi * (8_000 * t + 0.5 * 90_000 / t[-1] * t**2)
    sig = 1.1 * np.sin(2 * np.pi * 42_000 * t) + 0.28 * np.sin(chirp_phase)
    sig += 0.08 * rng.normal(size=t.size)
    write_scope_csv(out_dir / "stress_03_fft_chirp_spur.csv", {
        "TIME": t,
        "Signal_V": sig,
    }, {"Signal_V": "V"}, {"purpose": "FFT dominant spur with chirp background"})
    add("stress_03_fft_chirp_spur.csv", "FFT chirp plus fixed spur",
        ["fft"], _guide("fft", "Signal_V",
                        "A fixed 42 kHz component should dominate over the chirped background.",
                        {"f_min": 1000},
                        "Dominant frequency should be near 42 kHz."))

    # S04: filter with switching ripple and impulse-like EMI.
    t = np.linspace(0.0, 0.060, 24_000)
    env = 1800.0 * (1 - np.exp(-np.maximum(t - 0.004, 0) / 0.004))
    env *= np.exp(-np.maximum(t - 0.025, 0) / 0.080)
    noisy = env + 260.0 * np.sin(2 * np.pi * 230_000 * t) + 25.0 * rng.normal(size=t.size)
    noisy[::2400] += 900.0
    write_scope_csv(out_dir / "stress_04_filter_impulse_ringing.csv", {
        "TIME": t,
        "Noisy_current_A": noisy,
        "Envelope_A": env,
    }, {"Noisy_current_A": "A", "Envelope_A": "A"},
       {"purpose": "low-pass filter under impulse and ringing contamination"})
    add("stress_04_filter_impulse_ringing.csv", "Low-pass with impulses/ringing",
        ["lowpass", "fft"], _guide("lowpass", "Noisy_current_A",
                                   "Check that a 15 kHz low-pass suppresses switching ripple.",
                                   {"cutoff_hz": 15000, "label": "Noisy_current_A LP 15 kHz"}))

    # S05: calibration with offset/drift contamination.
    t = np.linspace(-0.005, 0.090, 6000)
    ref = rlc_pulse(t, 0.006, 0.004, 0.070, 2300.0)
    source = 0.72 * ref + 16.0 * np.sin(2 * np.pi * 9 * t) + rng.normal(0, 3.0, t.size)
    sensor_v = 2.5 + ref / 750.0 + 0.004 * rng.normal(size=t.size)
    write_scope_csv(out_dir / "stress_05_calibration_drift.csv", {
        "TIME": t,
        "Sensor_V": sensor_v,
        "Source_A_uncal": source,
        "Reference_A": ref,
    }, {"Sensor_V": "V", "Source_A_uncal": "A", "Reference_A": "A"},
       {"purpose": "formula and calibration under drift"})
    add("stress_05_calibration_drift.csv", "Calibration with drift",
        ["formula", "calibration"], _guide("calibration", "Source_A_uncal",
                                           "Fit a forced-origin gain to a reference while drift/noise are present.",
                                           {"reference": "Reference_A", "t_start": 0.010, "t_end": 0.075},
                                           "Slope should be close to 1/0.72 = 1.39."))

    # S06: hidden peak with trusted windows separated by an unreliable region.
    t = np.linspace(0.0, 0.180, 9000)
    true_i = rlc_pulse(t, 0.0025, 0.0022, 0.210, 8200.0)
    bbcm = np.minimum(true_i + rng.normal(0, 18.0, t.size), 6000.0)
    bbcm[(t > 0.008) & (t < 0.040)] = 6000.0
    pearson = true_i + rng.normal(0, 28.0, t.size)
    pearson[t > 0.007] *= 0.02
    write_scope_csv(out_dir / "stress_06_censored_multiwindow_6ka.csv", {
        "TIME": t,
        "BBCM_A": bbcm,
        "Pearson_A": pearson,
        "Control_V": control_pulse(t, 0.0, 0.150, 5.0),
    }, {"BBCM_A": "A", "Pearson_A": "A", "Control_V": "V"},
       {"purpose": "censored reconstruction with trusted windows"})
    add("stress_06_censored_multiwindow_6ka.csv", "Censored 6 kA multi-window recovery",
        ["saturation", "rlc"],
        _guide("rlc", "BBCM_A",
               "Recover a hidden peak using early Pearson data plus later clean BBCM windows.",
               {"sat_level": 6000, "ref_channel": "Pearson_A", "ref_start": 0,
                "ref_end": 0.006, "trusted_windows": "0:0.006, 0.050:0.150"},
               "RLC peak should exceed the 6 kA clip but remain physically plausible."))

    # S07: bipolar current and return energy.
    t = np.linspace(0.0, 0.120, 6000)
    pos = rlc_pulse(t, 0.005, 0.004, 0.050, 1800.0)
    neg = rlc_pulse(t, 0.055, 0.005, 0.045, 1300.0, sign=-1.0)
    cur = pos + neg + rng.normal(0, 5.0, t.size)
    write_scope_csv(out_dir / "stress_07_bipolar_return.csv", {
        "TIME": t,
        "Current_A": cur,
    }, {"Current_A": "A"}, {"purpose": "bipolar sign and integration stress"})
    add("stress_07_bipolar_return.csv", "Bipolar current return",
        ["stats", "integrate"], _guide("integrate", "Current_A",
                                       "Check sign-preserving statistics and cumulative current integral."))

    # S08: flatline/dropout and recovery.
    t = np.linspace(0.0, 0.090, 5000)
    sig = 420.0 * np.sin(2 * np.pi * 60 * t) + 15.0 * rng.normal(size=t.size)
    sig[(t > 0.035) & (t < 0.052)] = 0.0
    sig[(t > 0.071) & (t < 0.074)] = np.nan
    write_scope_csv(out_dir / "stress_08_flatline_dropout.csv", {
        "TIME": t,
        "Current_A": sig,
    }, {"Current_A": "A"}, {"purpose": "flatline dropout and NaN diagnostic stress"})
    add("stress_08_flatline_dropout.csv", "Flatline/dropout diagnostic",
        ["quality", "anomaly"], _guide("anomaly", "Current_A",
                                       "Find dropouts/NaNs and avoid confusing them with physics."))

    # S09: four modules with skew, imbalance, and one noisy channel.
    t = np.linspace(0.0, 0.100, 5000)
    base = rlc_pulse(t, 0.004, 0.003, 0.070, 1400.0)
    modules = {}
    for i, (gain, delay, noise) in enumerate([(1.0, 0.0, 7), (0.95, 0.0007, 8),
                                              (1.18, -0.0004, 9), (1.02, 0.0014, 25)], start=1):
        shifted = np.interp(t - delay, t, base, left=0.0, right=base[-1])
        modules[f"Module{i}_A"] = gain * shifted + rng.normal(0, noise, t.size)
    write_scope_csv(out_dir / "stress_09_module_skew_noise.csv", {"TIME": t, **modules},
                    {k: "A" for k in modules}, {"purpose": "module skew/imbalance/noise stress"})
    add("stress_09_module_skew_noise.csv", "Module skew and imbalance",
        ["stats", "anomaly"], _guide("stats", "Module3_A",
                                     "Compare modules; Module3 is intentionally high and Module4 noisy."))

    # S10: Rogowski-style derivative channel with integrator drift.
    t = np.linspace(-0.005, 0.080, 7000)
    current = rlc_pulse(t, 0.004, 0.003, 0.060, 2100.0)
    rogowski_v = 1e-4 * np.gradient(current, t) + 0.08 * t + 0.002 * rng.normal(size=t.size)
    write_scope_csv(out_dir / "stress_10_rogowski_drift.csv", {
        "TIME": t,
        "Current_A": current,
        "Rogowski_V": rogowski_v,
    }, {"Current_A": "A", "Rogowski_V": "V"},
       {"purpose": "Rogowski derivative/drift teaching stress"})
    add("stress_10_rogowski_drift.csv", "Rogowski drift/derivative",
        ["gradient", "integrate", "formula"], _guide("gradient", "Current_A",
                                                     "Compare dI/dt behavior against a Rogowski-style voltage channel."))

    # S11: high dynamic range left/right-axis plot.
    t = np.linspace(0.0, 0.160, 8000)
    current = rlc_pulse(t, 0.006, 0.006, 0.140, 6800.0)
    control = 0.15 * control_pulse(t, 0.006, 0.110, 1.0) + 0.01 * np.sin(2 * np.pi * 180 * t)
    write_scope_csv(out_dir / "stress_11_high_dynamic_range_axes.csv", {
        "TIME": t,
        "Current_A": current + rng.normal(0, 12.0, t.size),
        "Control_V": control,
    }, {"Current_A": "A", "Control_V": "V"}, {"purpose": "left/right axis high dynamic range"})
    add("stress_11_high_dynamic_range_axes.csv", "High dynamic range axes",
        ["stats", "lowpass"], _guide("stats", "Current_A",
                                     "Stress plot scaling with kA current and sub-volt control signal."))

    # S12: RL V/I/dI-dt sweep with internal inductance for surface/physics checks.
    t = np.linspace(0.0, 0.120, 6000)
    drive = np.where((t >= 0.005) & (t <= 0.080), 120.0, 0.0)
    current = np.zeros_like(t)
    dt = np.diff(t, prepend=t[0])
    for i in range(1, len(t)):
        didt_i = (drive[i - 1] - R_DEFAULT_OHM * current[i - 1]) / L_INTERNAL_H
        current[i] = current[i - 1] + didt_i * dt[i]
    current += 1.2 * np.sin(2 * np.pi * 22_000 * t)
    didt = np.gradient(current, t)
    write_scope_csv(out_dir / "stress_12_vi_didt_166uH_ripple.csv", {
        "TIME": t,
        "Drive_voltage_V": drive,
        "Current_A": current,
        "dIdt_A_per_s": didt,
        "L_dIdt_V": L_INTERNAL_H * didt,
    }, {"Drive_voltage_V": "V", "Current_A": "A", "dIdt_A_per_s": "A/s", "L_dIdt_V": "V"},
       {"purpose": "166 uH V/I/dI-dt stress with ripple"})
    add("stress_12_vi_didt_166uH_ripple.csv", "166 uH V/I/dI-dt ripple stress",
        ["gradient", "fft", "stats"], _guide("gradient", "Current_A",
                                             "Stress derivative and V=L*dI/dt consistency with ripple present."))

    (out_dir / "manifest.json").write_text(json.dumps({
        "generated_by": "scripts/generate_lite_stress_examples.py",
        "synthetic": True,
        "note": "Advanced stress data for tool fault-finding; not experimental truth.",
        "default_internal_inductance_H": L_INTERNAL_H,
        "datasets": manifest,
    }, indent=2) + "\n", encoding="utf-8")

    readme = [
        "# Scope Analyzer Lite stress-test examples",
        "",
        "These CSVs are synthetic fault-finding cases for the deterministic Lite",
        "toolbox. They are intentionally more difficult than the beginner benchmark",
        "examples: larger traces, bad timestamps, clipped/censored peaks, NaNs,",
        "dropouts, high dynamic range, and drift.",
        "",
        "They are safe to regenerate. Original user CSV files are never modified.",
        "",
        "Dataset index:",
    ]
    for entry in manifest:
        readme.append(f"- {entry['id']} {entry['file']}: {entry['title']} -> {', '.join(entry['tools'])}")
    readme.extend([
        "",
        "Regenerate with: python scripts/generate_lite_stress_examples.py",
        "Benchmark with: python scripts/benchmark_lite_stress_tools.py",
    ])
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    items = make_stress_examples(DEFAULT_OUT)
    print(f"Generated {len(items)} stress-test CSV files in {DEFAULT_OUT}")
