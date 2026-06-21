# Scope Analyzer Lite Toolbox FAQ

Scope Analyzer Lite is designed so the useful scientific tools work even when no
LLM is installed. The LLM is optional; the numbers come from deterministic
NumPy/SciPy tools.

Core safety rule: the original CSV is read-only. Every formula, filter,
smoothing pass, derivative, integral, reconstruction, and overlay is an in-memory
display result unless you explicitly export a new figure or derived file.

## Fast Start

1. Click `Examples -> 02 - BBCM clipped hidden peak` in the top toolbar.
2. Choose `Tools & libraries -> Recover hidden peak / RLC reconstruction`.
3. Select `BBCM_A`.
4. Enter saturation level `6000`.
5. Enter trusted windows `0:0.005, 0.040:0.150`.
6. Press `Run`.
7. Read the text report and inspect the overlay. The overlay is a model estimate,
   not measured data.

Fallback: after the first packaged-app launch, the same CSVs are mirrored to
`~/Documents/Scope Analyzer/examples/tool_benchmarks/`, so they can also be
opened with `Open CSV`.

## In-App Examples Menu

The `Examples` dropdown is the recommended way to audit the Lite toolbox. It
loads the same 15 benchmark CSVs that the automated test suite uses, so a visual
check and a scripted check are looking at the same cases. Each menu item lists
the main tools to try on that dataset.

## Included Benchmark Datasets

The full benchmark pack is under `examples/tool_benchmarks/`.

| File | Main Lesson | Try This Tool |
| --- | --- | --- |
| `01_clean_rl_pulse.csv` | Clean overdamped pulse | Statistics, derivative, integral, RLC |
| `02_bbcm_clipped_6ka.csv` | 6 kA clipped BBCM hidden peak | Saturation estimate, RLC, Analyze shot |
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
- Expected: a reconstructed RLC overlay and a peak estimate in the hidden region.

Interpretation rule: the reconstruction is a model estimate. If the overlay misses
clean measured regions, the circuit model is missing dynamics such as snubbers,
switch behavior, or magnetic/core effects.

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

Example:
- Open `02_bbcm_clipped_6ka.csv`.
- Run `Analyze shot pipeline` on `BBCM_A`.
- Expected: statistics, anomaly scan, saturation estimate, and hidden-peak/RLC
  reconstruction in one report.

## How To Run The Benchmark

From the project folder:

```bash
source venv/bin/activate
python scripts/generate_lite_toolbox_examples.py
python scripts/benchmark_lite_toolbox.py
```

Expected result: all 15 synthetic benchmark cases pass. The report is written to
`backtests/lite_toolbox_benchmark.txt`.
