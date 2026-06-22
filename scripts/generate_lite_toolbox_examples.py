"""Generate deterministic Lite toolbox benchmark CSV files.

These examples are educational/reference data for Scope Analyzer Lite. They
exercise the no-LLM deterministic tools exposed through scope_web/backend_api.py:
stats, QC, formula calibration, filtering, smoothing, derivative, integral,
FFT, anomaly detection, saturation recovery, RLC reconstruction, reference
calibration, and the one-click pipeline.

All files are synthetic demonstration data, not experimental truth.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "examples" / "tool_benchmarks"
L_INTERNAL_H = 166e-6
R_DEFAULT_OHM = 0.055


def rlc_pulse(t: np.ndarray, t0: float, tau_r: float, tau_d: float,
              peak: float, sign: float = 1.0) -> np.ndarray:
    y = np.zeros_like(t, dtype=np.float64)
    m = t > t0
    dt = t[m] - t0
    shape = np.exp(-dt / tau_d) - np.exp(-dt / tau_r)
    if shape.size and np.nanmax(np.abs(shape)) > 0:
        shape *= peak / np.nanmax(shape)
    y[m] = sign * shape
    return y


def control_pulse(t: np.ndarray, start: float, end: float,
                  amplitude: float = 5.0) -> np.ndarray:
    return np.where((t >= start) & (t <= end), amplitude, 0.0)


def write_scope_csv(path: Path, columns: dict[str, np.ndarray],
                    units: dict[str, str] | None = None,
                    meta: dict[str, str] | None = None) -> None:
    units = units or {}
    meta = meta or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns.keys())
    n = len(next(iter(columns.values())))
    with path.open("w", newline="", encoding="utf-8") as handle:
        wr = csv.writer(handle)
        wr.writerow(["Model", "Scope Studio synthetic benchmark"])
        wr.writerow(["Purpose", meta.get("purpose", "Lite toolbox validation")])
        wr.writerow(["Synthetic", "true"])
        wr.writerow(["Source", *names[1:]])
        wr.writerow(["Vertical Units", *[units.get(name, "") for name in names[1:]]])
        wr.writerow(names)
        for i in range(n):
            row = []
            for name in names:
                v = columns[name][i]
                if isinstance(v, float) and math.isnan(v):
                    row.append("NaN")
                else:
                    row.append(f"{float(v):.10e}")
            wr.writerow(row)


def make_examples(out_dir: Path = DEFAULT_OUT) -> list[dict[str, object]]:
    rng = np.random.default_rng(20260621)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    def add(entry: dict[str, object]) -> None:
        manifest.append(entry)

    # 1. Clean overdamped RLC-like pulse.
    t = np.linspace(0.0, 0.120, 4000)
    cur = rlc_pulse(t, 0.005, 0.003, 0.090, 4200.0)
    didt = np.gradient(cur, t)
    vl = L_INTERNAL_H * didt
    write_scope_csv(out_dir / "01_clean_rl_pulse.csv", {
        "TIME": t,
        "Current_A": cur,
        "Control_V": control_pulse(t, 0.005, 0.080),
        "L_dIdt_V": vl,
    }, {"Current_A": "A", "Control_V": "V", "L_dIdt_V": "V"},
       {"purpose": "clean RLC pulse for stats, derivative, integral, RLC"})
    add({"id": "01", "file": "01_clean_rl_pulse.csv", "title": "Clean RLC pulse", "tools": ["stats", "gradient", "integrate", "rlc", "pipeline"]})

    # 2. BBCM-style clipped 6 kA hidden peak with Pearson early reference.
    t = np.linspace(0.0, 0.180, 6000)
    true_i = rlc_pulse(t, 0.0018, 0.0020, 0.220, 7350.0)
    bbcm_noise = true_i + rng.normal(0.0, 18.0, t.size)
    bbcm = np.where(true_i >= 6000.0, 6000.0, bbcm_noise)
    pearson = true_i + rng.normal(0.0, 25.0, t.size)
    pearson[t > 0.012] = pearson[t > 0.012] * 0.03
    write_scope_csv(out_dir / "02_bbcm_clipped_6ka.csv", {
        "TIME": t,
        "BBCM_A": bbcm,
        "Pearson_A": pearson,
        "Control_V": control_pulse(t, 0.0, 0.150, 5.0),
    }, {"BBCM_A": "A", "Pearson_A": "A", "Control_V": "V"},
       {"purpose": "6 kA clipping benchmark for saturation and RLC reconstruction"})
    add({"id": "02", "file": "02_bbcm_clipped_6ka.csv", "title": "BBCM clipped hidden peak", "tools": ["saturation", "rlc", "rlc_audit", "pipeline"]})

    # 3. Low-pass ringing example.
    t = np.linspace(0.0, 0.040, 5000)
    base = 1200.0 * (1.0 - np.exp(-np.maximum(t - 0.003, 0.0) / 0.004))
    base *= np.exp(-np.maximum(t - 0.015, 0.0) / 0.040)
    ringing = 220.0 * np.exp(-np.maximum(t - 0.003, 0.0) / 0.010) * np.sin(2 * np.pi * 150_000 * t)
    noisy = base + ringing + rng.normal(0.0, 28.0, t.size)
    write_scope_csv(out_dir / "03_lowpass_ringing.csv", {
        "TIME": t,
        "Current_noisy_A": noisy,
        "Current_clean_A": base,
    }, {"Current_noisy_A": "A", "Current_clean_A": "A"},
       {"purpose": "15 kHz low-pass removes 150 kHz ringing/noise"})
    add({"id": "03", "file": "03_lowpass_ringing.csv", "title": "Ringing/noise low-pass", "tools": ["lowpass", "fft", "anomaly"]})

    # 4. FFT two-tone signal, dominant 20 kHz.
    fs = 1_000_000.0
    t = np.arange(0, 0.020, 1 / fs)
    sig = 1.2 * np.sin(2 * np.pi * 20_000 * t) + 0.35 * np.sin(2 * np.pi * 80_000 * t)
    sig += 0.04 * rng.normal(size=t.size)
    write_scope_csv(out_dir / "04_fft_two_tone.csv", {
        "TIME": t,
        "Signal_V": sig,
    }, {"Signal_V": "V"}, {"purpose": "FFT dominant-frequency benchmark"})
    add({"id": "04", "file": "04_fft_two_tone.csv", "title": "FFT two-tone", "tools": ["fft", "lowpass"]})

    # 5. Formula and forced-origin calibration pair.
    t = np.linspace(0.0, 0.080, 3000)
    ref = rlc_pulse(t, 0.004, 0.003, 0.070, 1800.0)
    sensor_v = 2.5 + ref / 750.0 + rng.normal(0.0, 0.002, t.size)
    sensor_scaled = 0.82 * ref + rng.normal(0.0, 2.5, t.size)
    write_scope_csv(out_dir / "05_calibration_pair.csv", {
        "TIME": t,
        "Sensor_V": sensor_v,
        "Sensor_scaled_A": sensor_scaled,
        "Reference_A": ref,
    }, {"Sensor_V": "V", "Sensor_scaled_A": "A", "Reference_A": "A"},
       {"purpose": "formula conversion and forced-origin reference calibration"})
    add({"id": "05", "file": "05_calibration_pair.csv", "title": "Formula/calibration pair", "tools": ["formula", "calibration", "stats"]})

    # 6. Derivative and inductive voltage from 166 uH.
    t = np.linspace(0.0, 0.050, 3000)
    cur = np.piecewise(t, [t < 0.006, (t >= 0.006) & (t < 0.020), (t >= 0.020) & (t < 0.038), t >= 0.038],
                       [0.0, lambda x: 900.0 * (x - 0.006) / 0.014, 900.0, lambda x: 900.0 * np.maximum(1 - (x - 0.038) / 0.010, 0.0)])
    didt = np.gradient(cur, t)
    write_scope_csv(out_dir / "06_didt_voltage_166uH.csv", {
        "TIME": t,
        "Current_A": cur,
        "dIdt_A_per_s_true": didt,
        "L_dIdt_V": L_INTERNAL_H * didt,
    }, {"Current_A": "A", "dIdt_A_per_s_true": "A/s", "L_dIdt_V": "V"},
       {"purpose": "dI/dt and L*dI/dt relation for L=166 uH"})
    add({"id": "06", "file": "06_didt_voltage_166uH.csv", "title": "dI/dt and inductive voltage", "tools": ["gradient", "stats"]})

    # 7. Charge/integral reference pulse.
    t = np.linspace(0.0, 0.100, 3000)
    cur = np.where((t >= 0.010) & (t <= 0.060), 100.0, 0.0)
    edge = np.exp(-((t - 0.010) / 0.0012) ** 2) - np.exp(-((t - 0.060) / 0.0012) ** 2)
    cur = cur + 8.0 * edge
    write_scope_csv(out_dir / "07_charge_integral.csv", {
        "TIME": t,
        "Current_A": cur,
    }, {"Current_A": "A"}, {"purpose": "integral/charge benchmark near 5 C"})
    add({"id": "07", "file": "07_charge_integral.csv", "title": "Charge integral", "tools": ["integrate", "stats"]})

    # 8. Moving-average noise reduction.
    t = np.linspace(0.0, 0.080, 3500)
    clean = np.where((t >= 0.010) & (t <= 0.060), 500.0, 0.0)
    noisy = clean + 55.0 * rng.normal(size=t.size)
    write_scope_csv(out_dir / "08_moving_average_noise.csv", {
        "TIME": t,
        "Noisy_current_A": noisy,
        "Clean_current_A": clean,
    }, {"Noisy_current_A": "A", "Clean_current_A": "A"},
       {"purpose": "moving average noise reduction"})
    add({"id": "08", "file": "08_moving_average_noise.csv", "title": "Moving average noise", "tools": ["movmean", "stats"]})

    # 9. Sparse spikes/anomalies.
    t = np.linspace(0.0, 0.060, 4000)
    cur = 250.0 * np.sin(2 * np.pi * 120 * t) + 20.0 * rng.normal(size=t.size)
    for center, amp in [(0.014, 1800.0), (0.031, -1400.0), (0.047, 1200.0)]:
        idx = np.argmin(np.abs(t - center))
        cur[idx:idx + 4] += amp
    write_scope_csv(out_dir / "09_spikes_anomalies.csv", {
        "TIME": t,
        "Current_A": cur,
    }, {"Current_A": "A"}, {"purpose": "robust anomaly/spike detection"})
    add({"id": "09", "file": "09_spikes_anomalies.csv", "title": "Spike anomalies", "tools": ["anomaly"]})

    # 10. CSV quality: duplicate/backwards timestamp, large gap, NaN data.
    t = np.linspace(0.0, 0.020, 1200)
    t[400] = t[399]
    t[700] = t[699] - 2e-5
    t[900:] += 0.004
    y = np.sin(2 * np.pi * 500 * t)
    y[500] = np.nan
    write_scope_csv(out_dir / "10_quality_gap_nan_duplicate.csv", {
        "TIME": t,
        "Signal_V": y,
    }, {"Signal_V": "V"}, {"purpose": "data-quality warnings"})
    add({"id": "10", "file": "10_quality_gap_nan_duplicate.csv", "title": "Bad timing/NaN quality case", "tools": ["quality"]})

    # 11. Baseline offset and pre-trigger correction.
    t = np.linspace(-0.010, 0.050, 3000)
    pulse = rlc_pulse(t, 0.002, 0.0025, 0.045, 950.0)
    raw = pulse + 42.0 + 3.0 * rng.normal(size=t.size)
    write_scope_csv(out_dir / "11_baseline_offset.csv", {
        "TIME": t,
        "Raw_offset_A": raw,
        "True_current_A": pulse,
    }, {"Raw_offset_A": "A", "True_current_A": "A"},
       {"purpose": "formula helper baseline(x,t,end) demonstration"})
    add({"id": "11", "file": "11_baseline_offset.csv", "title": "Baseline-offset formula", "tools": ["formula", "stats"]})

    # 12. Soft saturation: compressed but not flat above 6 kA.
    t = np.linspace(0.0, 0.160, 5000)
    true_i = rlc_pulse(t, 0.002, 0.0022, 0.160, 7600.0)
    sat = 6000.0
    soft = np.where(true_i <= sat, true_i, sat + 0.18 * (true_i - sat))
    soft += rng.normal(0.0, 14.0, t.size)
    write_scope_csv(out_dir / "12_soft_saturation.csv", {
        "TIME": t,
        "Soft_BBCM_A": soft,
        "True_current_A": true_i,
    }, {"Soft_BBCM_A": "A", "True_current_A": "A"},
       {"purpose": "known-level soft saturation benchmark"})
    add({"id": "12", "file": "12_soft_saturation.csv", "title": "Soft saturation", "tools": ["saturation", "rlc"]})

    # 13. Four-module balance example.
    t = np.linspace(0.0, 0.100, 3000)
    base = rlc_pulse(t, 0.004, 0.004, 0.080, 1550.0)
    modules = {
        "Module1_A": base * 1.00 + rng.normal(0, 8, t.size),
        "Module2_A": base * 0.98 + rng.normal(0, 8, t.size),
        "Module3_A": base * 1.16 + rng.normal(0, 8, t.size),
        "Module4_A": base * 1.02 + rng.normal(0, 8, t.size),
    }
    write_scope_csv(out_dir / "13_module_balance.csv", {"TIME": t, **modules},
                    {k: "A" for k in modules}, {"purpose": "module-balance teaching dataset"})
    add({"id": "13", "file": "13_module_balance.csv", "title": "Four-module balance", "tools": ["stats", "anomaly"]})

    # 14. Negative pulse reconstruction sign-normalization.
    t = np.linspace(0.0, 0.120, 4000)
    neg = rlc_pulse(t, 0.004, 0.003, 0.100, 2600.0, sign=-1.0)
    write_scope_csv(out_dir / "14_negative_pulse.csv", {
        "TIME": t,
        "Negative_current_A": neg,
    }, {"Negative_current_A": "A"}, {"purpose": "negative-polarity pulse statistics/RLC"})
    add({"id": "14", "file": "14_negative_pulse.csv", "title": "Negative current pulse", "tools": ["stats", "rlc"]})

    # 15. Synthetic V/I/dI-dt with 166 uH internal inductance.
    t = np.linspace(0.0, 0.120, 5000)
    vdrive = np.where((t >= 0.005) & (t <= 0.080), 80.0, 0.0)
    current = np.zeros_like(t)
    dt = np.diff(t, prepend=t[0])
    for i in range(1, len(t)):
        didt_i = (vdrive[i - 1] - R_DEFAULT_OHM * current[i - 1]) / L_INTERNAL_H
        current[i] = current[i - 1] + didt_i * dt[i]
    current += 0.8 * np.sin(2 * np.pi * 18_000 * t)
    didt = np.gradient(current, t)
    write_scope_csv(out_dir / "15_vi_didt_166uH.csv", {
        "TIME": t,
        "Drive_voltage_V": vdrive,
        "Current_A": current,
        "dIdt_A_per_s": didt,
        "L_dIdt_V": L_INTERNAL_H * didt,
    }, {"Drive_voltage_V": "V", "Current_A": "A", "dIdt_A_per_s": "A/s", "L_dIdt_V": "V"},
       {"purpose": "RL V/I/dI-dt teaching benchmark, L=166 uH"})
    add({"id": "15", "file": "15_vi_didt_166uH.csv", "title": "V/I/dI-dt 166 uH", "tools": ["gradient", "fft", "stats"]})

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "generated_by": "scripts/generate_lite_toolbox_examples.py",
        "synthetic": True,
        "note": "Educational benchmark data only; not experimental truth.",
        "default_internal_inductance_H": L_INTERNAL_H,
        "datasets": manifest,
    }, indent=2) + "\n", encoding="utf-8")

    readme = [
        "# Scope Analyzer Lite toolbox benchmark examples",
        "",
        "These 15 CSV files are synthetic reference shots for testing and teaching the",
        "Lite deterministic toolbox. They are designed to be opened with Open CSV in",
        "Scope Analyzer Lite. The original CSV is read-only; formulas, filters,",
        "reconstructions, and overlays are display/in-memory operations.",
        "",
        "Recommended first tour:",
        "1. Open 02_bbcm_clipped_6ka.csv and run Recover hidden peak / RLC reconstruction with sat_level=6000.",
        "2. Open 03_lowpass_ringing.csv and apply a 15 kHz low-pass filter.",
        "3. Open 05_calibration_pair.csv and apply formula (x-2.5)*750 to Sensor_V.",
        "4. Open 04_fft_two_tone.csv and run FFT; the dominant tone should be near 20 kHz.",
        "5. Open 15_vi_didt_166uH.csv to connect V=L*dI/dt with the 166 uH teaching model.",
        "",
        "Dataset index:",
    ]
    for entry in manifest:
        readme.append(f"- {entry['id']} {entry['file']}: {entry['title']} -> {', '.join(entry['tools'])}")
    readme.append("")
    readme.append("Regenerate with: python scripts/generate_lite_toolbox_examples.py")
    readme.append("Benchmark with: python scripts/benchmark_lite_toolbox.py")
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    items = make_examples(DEFAULT_OUT)
    print(f"Generated {len(items)} synthetic Lite toolbox CSV files in {DEFAULT_OUT}")
