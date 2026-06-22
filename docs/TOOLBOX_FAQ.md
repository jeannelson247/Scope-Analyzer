# Scope Analyzer Lite Toolbox FAQ

Scope Analyzer Lite is designed so the useful scientific tools work even when no
LLM is installed. The LLM is optional; the numbers come from deterministic
NumPy/SciPy tools.

Core safety rule: the original CSV is read-only. Every formula, filter,
smoothing pass, derivative, integral, reconstruction, and overlay is an in-memory
display result unless you explicitly export a new figure or derived file.
The `Export analyzed CSV` button always writes a new CSV plus a metadata sidecar;
it never overwrites the source oscilloscope file.

## Fast Start

1. Click `Examples -> 02 - BBCM clipped hidden peak` in the top toolbar.
2. Read the `Example loaded` panel.
3. Press `Run suggested tool`.
4. Read the text report and inspect the overlay. The overlay is a model estimate,
   not measured data.

Manual path, if you want to set the parameters yourself:

1. Choose `Tools & libraries -> Recover hidden peak / RLC reconstruction`.
2. Select `BBCM_A`.
3. Enter saturation level `6000`.
4. Enter trusted windows `0:0.005, 0.040:0.150`.
5. Press `Run`.

Fallback: after the first packaged-app launch, the same CSVs are mirrored to
`~/Documents/Scope Analyzer/examples/tool_benchmarks/`, so they can also be
opened with `Open CSV/TXT`. If a developer checkout or packaged build is missing
the generated examples, Lite regenerates the synthetic benchmark pack there on
demand.

The menu also includes an advanced `Stress-test datasets` section when those
files are available. These are deliberately rougher than the beginner examples:
larger traces, bad timestamps, NaNs, hidden clipping, dropouts, module skew,
Rogowski-style drift, high dynamic range, and V/I/dI-dt ripple. They are for
fault-finding before release and for checking tools by eye.

## Loading Your Own Oscilloscope CSV/TXT/TSV

Click `Open CSV/TXT` and choose the file exported by the scope. Lite immediately
shows a `CSV import report` with the loader decisions it made:

- delimiter (`comma`, `semicolon`, or `tab`),
- number of skipped preamble/header rows,
- detected scope/model when the preamble provides it,
- detected time column and signal columns,
- detected channel units,
- sample interval / sample rate from the time column,
- QC status and first issue, if any,
- confirmation that the source CSV remains read-only.

If the report says `QC WARNING` or `QC ERROR`, press `Run CSV quality report`
before doing calibration, filtering, FFT, integration, or reconstruction. The
quality report is deterministic and does not modify the CSV.

## In-App Examples Menu

The `Examples` dropdown is the recommended way to audit the Lite toolbox. It
loads the same 15 benchmark CSVs that the automated test suite uses, so a visual
check and a scripted check are looking at the same cases. Each menu item lists
the main tools to try on that dataset. After an example loads, Lite opens a small
`Example loaded` guide with:

- the recommended deterministic tool,
- the column to analyze,
- any useful default parameters,
- what result to expect, and
- a `Run suggested tool` button that pre-fills the tool panel.

## Included Benchmark Datasets

The full benchmark pack is under `examples/tool_benchmarks/` in a developer
checkout, and under `~/Documents/Scope Analyzer/examples/tool_benchmarks/` after
the packaged app launches. The pack is synthetic and deterministic, so it can be
regenerated safely if missing.

| File | Main Lesson | Try This Tool |
| --- | --- | --- |
| `01_clean_rl_pulse.csv` | Clean overdamped pulse | Statistics, derivative, integral, RLC |
| `02_bbcm_clipped_6ka.csv` | 6 kA clipped BBCM hidden peak | Saturation estimate, RLC, Reconstruction audit, Analyze shot |
| `03_lowpass_ringing.csv` | High-frequency ringing/noise | Low-pass filter, FFT, anomaly scan |
| `04_fft_two_tone.csv` | Dominant frequency detection | FFT |
| `05_calibration_pair.csv` | Formula conversion and reference calibration | Formula builder, calibration |
| `06_didt_voltage_166uH.csv` | `V = L*dI/dt` with `L = 166 uH` | Derivative, statistics |
| `07_charge_integral.csv` | Current-to-charge integral | Integral |
| `08_moving_average_noise.csv` | Smoothing noisy plateau current | Moving average |
| `09_spikes_anomalies.csv` | Sparse EMI/spike events | Anomaly scan |
| `10_quality_gap_nan_duplicate.csv` | Bad timestamps and NaN values | CSV quality report |
| `11_baseline_offset.csv` | Pre-trigger baseline subtraction | Formula builder |
| `12_soft_saturation.csv` | Soft monitor compression above known level | Saturation, RLC |
| `13_module_balance.csv` | Comparing module currents | Statistics |
| `14_negative_pulse.csv` | Negative-polarity current | Statistics, RLC |
| `15_vi_didt_166uH.csv` | RL drive/current/dI-dt teaching model | Derivative, FFT, statistics |

## Advanced Stress-Test Datasets

The stress pack is under `examples/tool_stress/` in a developer checkout and
under `~/Documents/Scope Analyzer/examples/tool_stress/` after packaged-app
launch. These files are synthetic and deterministic. They are intended to make
tool failures obvious before the app is shared.

| File | Main Stress | Try This Tool |
| --- | --- | --- |
| `stress_01_large_decimation_spikes.csv` | Larger trace plus sparse spikes | Anomaly scan, statistics, low-pass |
| `stress_02_nonuniform_time_nan.csv` | Duplicate/backward time, gap, NaNs | CSV quality report |
| `stress_03_fft_chirp_spur.csv` | Chirp plus fixed 42 kHz spur | FFT |
| `stress_04_filter_impulse_ringing.csv` | Impulses plus 230 kHz ringing | Low-pass, FFT |
| `stress_05_calibration_drift.csv` | Reference calibration with drift/noise | Formula, reference calibration |
| `stress_06_censored_multiwindow_6ka.csv` | 6 kA censoring with separated trusted windows | Saturation, RLC reconstruction |
| `stress_07_bipolar_return.csv` | Positive and negative current pulse | Statistics, integral |
| `stress_08_flatline_dropout.csv` | Flatline and NaN dropout | CSV quality, anomaly scan |
| `stress_09_module_skew_noise.csv` | Four modules with skew/imbalance/noise | Statistics, anomaly scan |
| `stress_10_rogowski_drift.csv` | Rogowski-style derivative channel with drift | Derivative, integral |
| `stress_11_high_dynamic_range_axes.csv` | kA current with sub-volt control signal | Statistics, low-pass |
| `stress_12_vi_didt_166uH_ripple.csv` | `L = 166 uH` V/I/dI-dt with ripple | Derivative, FFT, statistics |

Developer commands:

```bash
python scripts/generate_lite_stress_examples.py
python scripts/benchmark_lite_stress_tools.py
```

## Tool Recipes

### Statistics

Use for publication captions and quick sanity checks.

Example:
- Open `01_clean_rl_pulse.csv`.
- Select `Current_A`.
- Run `Tools & libraries -> Statistics`.
- Expected: peak current near 4.2 kA, plus min, mean, RMS, median, p5, p95, and sample count.

### CSV Quality Report

Use before trusting any analysis. It checks timing, duplicate/backwards samples,
large gaps, and NaN/nonfinite values.

Example:
- Open `10_quality_gap_nan_duplicate.csv`.
- Run `CSV quality report`.
- Expected: warning or error status because the file has duplicate/backwards time
  steps, a large gap, and one NaN data value.

### Formula / Preset Builder

Use to convert monitor voltage into physical units without changing the source
CSV.

Example:
- Open `05_calibration_pair.csv`.
- Select `Sensor_V`.
- Formula: `(x-2.5)*750`.
- Label: `Sensor_V_to_A`.
- Unit: `A`.
- Expected: a new display trace close to `Reference_A`.

Useful formulas:
- BBCM 750 A/V centered at 2.5 V: `(x-2.5)*750`.
- BBCM inverted form: `(2.5-x)*750`.
- Baseline correction using pre-trigger data: `baseline(x,t,-0.001)`.
- Moving average inside a formula: `movmean(x,101)`.
- Derivative inside a formula: `gradient(x,t)`.

### Low-Pass Filter

Use to remove ringing/noise while preserving the slower current envelope.

Example:
- Open `03_lowpass_ringing.csv`.
- Select `Current_noisy_A`.
- Enter cutoff `15000` Hz, or use the UI control as `15 kHz`.
- Expected: a derived low-pass trace with much less 150 kHz ringing.

### Moving Average

Use for quick smoothing when a physics filter is not needed.

Example:
- Open `08_moving_average_noise.csv`.
- Select `Noisy_current_A`.
- Window: `101` samples.
- Expected: a smoother plateau trace.

### Derivative / dI/dt

Use for rise-rate analysis or inductive-voltage estimates.

Example:
- Open `06_didt_voltage_166uH.csv`.
- Run derivative on `Current_A`.
- Compare with `dIdt_A_per_s_true` and `L_dIdt_V`.
- Physics relation: `V_L = L*dI/dt`, with `L = 166e-6 H` in this benchmark.

### Integral

Use to estimate charge or time-integrated current.

Example:
- Open `07_charge_integral.csv`.
- Run integral on `Current_A`.
- Expected: final charge near 5 C because the current is about 100 A for about 50 ms.

### Export Analyzed CSV

Use when you want to share or archive the exact traces currently selected in the
Lite workbench. This is different from `Export figure (PNG)`: the figure export
saves the plot image, while analyzed CSV export saves selected raw and derived
traces as numeric data.

Example:
- Load any CSV or benchmark example.
- Apply a formula, low-pass filter, derivative, integral, or moving average.
- Tick the channels you want to include.
- Press `Export analyzed CSV`.
- Choose a save location.

Expected: Lite writes a new `*_analyzed.csv` file and a matching
`*.meta.json` sidecar describing the source file, selected columns, and any
in-memory transforms. The original oscilloscope CSV is not modified.

### FFT / Dominant Frequency

Use to identify ringing, switching ripple, or noise pickup.

Example:
- Open `04_fft_two_tone.csv`.
- Run FFT on `Signal_V` with minimum frequency `1000` Hz.
- Expected: dominant frequency near 20 kHz.

### Anomaly Scan

Use to find spikes, clipping, dropouts, baseline drift, and high crest factor.

Example:
- Open `09_spikes_anomalies.csv`.
- Run anomaly scan on `Current_A` with threshold sigma `5`.
- Expected: the report flags injected spike events.

### Saturation Estimate

Use when a current monitor has clipped or compressed above a known level.

Example:
- Open `02_bbcm_clipped_6ka.csv`.
- Select `BBCM_A`.
- Saturation level: `6000`.
- Expected: the tool marks the saturated interval and estimates the hidden peak.

For soft saturation, use the known saturation level even if the trace does not
look perfectly flat. Try `12_soft_saturation.csv` with level `6000`.

### Recover Hidden Peak / RLC Reconstruction

Use when part of the measured pulse is censored but clean windows still constrain
the circuit model. This is more appropriate than a simple visible-data fit when
the sensor is saturated.

Example:
- Open `02_bbcm_clipped_6ka.csv`.
- Select `BBCM_A`.
- Saturation level: `6000`.
- Reference: `Pearson_A`.
- Reference window: `0` to `0.010` seconds.
- Trusted target windows: `0:0.005, 0.040:0.150`.
- Optional physical hints: enter measured `R` in ohms, `L` in microhenries in
  the Lite UI, `C` in farads, and initial capacitor charging voltage `V0` in
  volts. The Python bridge receives SI units and compares the fit to
  `tau_rise = L/R`, `tau_droop = R*C`, and initial slope
  `dI/dt(0+) ~= V0/L`.
- Soft prior: leave at `0` for report-only comparison, or use a small value
  such as `0.1` to `0.3` when you want the physical rig values to gently
  regularize the fitted time constants.
- Expected: a reconstructed RLC overlay and a peak estimate in the hidden region.

Interpretation rule: the reconstruction is a model estimate. If the overlay misses
clean measured regions, the circuit model is missing dynamics such as snubbers,
switch behavior, or magnetic/core effects.

Optional inputs that improve reconstruction accuracy:

- `Saturation level`: known sensor limit, e.g. BBCM clip level in amperes.
- `Trusted target windows`: time regions where the target monitor is known to be
  accurate, such as before saturation and after recovery.
- `Reference channel/window`: a Pearson/Rogowski/channel that is reliable over a
  limited window and measures the same current.
- `R`: total effective series resistance during the discharge, including busbar,
  switch, cable, capacitor ESR, and contact resistance when known.
- `L`: effective loop/internal inductance in henries; the Lite UI accepts `uH`.
- `C`: measured usable capacitance in farads, not just nameplate capacitance.
- `V0`: initial charging voltage immediately before the pulse.
- `Switch-on/switch-off times`: use fit windows to avoid forcing a free-discharge
  RLC model across active switching or forced turn-off regions.
- `Polarity/sign`: use the channel calibration formula/gain so current polarity
  is physically meaningful before reconstruction.
- `Filter cutoff`: use the same deterministic low-pass setting you trust for
  the measurement; do not filter so aggressively that the real rise is erased.

### Reconstruction Audit

Use after RLC reconstruction when you need to decide whether the overlay is
scientifically trustworthy or merely a plausible-looking model curve.

The audit compares:

- Saturation extrapolation peak estimates.
- Censored-RLC reconstructed peak.
- Agreement between peak-estimation methods.
- Clean-window residuals.
- `tau_rise` versus `L/R` when `R` and `L` are supplied.
- `tau_droop` versus `R*C` when `R` and `C` are supplied.
- Initial `dI/dt` versus `V0/L` when `V0` and `L` are supplied.
- Optional sensitivity of the reconstructed peak to small changes in `R`, `L`,
  `C`, and `V0`.

Verdicts:

- `Reliable with current assumptions`: residuals, method agreement, and
  physical hints are mutually consistent.
- `Plausible but model mismatch`: the overlay may be useful, but at least one
  assumption check is weak. This is common when the apparent `tau_droop` differs
  from `R*C`, suggesting missing switch/snubber/ESR/contact/core dynamics.
- `Do not trust yet`: the clean-data residual, method disagreement, or physical
  mismatch is too large for a defensible reconstruction.

Example:

- Open `02_bbcm_clipped_6ka.csv`.
- Run `Reconstruction audit`.
- Use the same saturation/reference/trusted windows as the RLC tool.
- Optional: set `R`, `L`, `C`, `V0`, prior weight `0.1` to `0.3`, and
  sensitivity sweep `10%`.
- Expected: a verdict, method-comparison table, physical consistency ratios,
  recommendations, and an RLC overlay.

### Reference Calibration

Use to fit a forced-origin gain between a source channel and a reference channel.

Example:
- Open `05_calibration_pair.csv`.
- Source: `Sensor_scaled_A`.
- Reference: `Reference_A`.
- Fit start/end: `0.006` to `0.070` seconds.
- Expected: slope near `1.22`, because the source was synthesized as about
  `0.82 * Reference_A`.

### Analyze Shot Pipeline

Use when you want a first-pass report without deciding which tool to run.
In the Lite app, `Analyze shot` now opens a guided panel where you can choose
which deterministic steps run:

- Statistics.
- Anomaly scan.
- Saturation estimate.
- Hidden-peak / RLC reconstruction.
- Reconstruction audit / trust verdict.

The same panel accepts saturation level, reference channel/window, trusted
target windows, and optional measured `R`, `L`, `C`, and `V0` values for the RLC
reconstruction. If no advanced fields are filled, it still behaves like the
simple one-click pipeline.

Example:
- Open `02_bbcm_clipped_6ka.csv`.
- Run `Analyze shot pipeline` on `BBCM_A`.
- To focus only on reconstruction, untick statistics/anomaly/saturation and
  leave `RLC reconstruction` and/or `Reconstruction audit` ticked.
- Expected: selected deterministic analyses in one report, with original CSV
  unchanged.

## How To Run The Benchmark

From the project folder:

```bash
source venv/bin/activate
python scripts/generate_lite_toolbox_examples.py
python scripts/benchmark_lite_toolbox.py
```

Expected result: all 15 synthetic benchmark cases pass. The report is written to
`backtests/lite_toolbox_benchmark.txt`.
