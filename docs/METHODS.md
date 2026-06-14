# Scope Studio Methods

This document states what Scope Studio computes and what it does not claim.
Raw oscilloscope CSV files are treated as immutable measurements. Display
transforms, overlays, and exported analyzed copies are derived artifacts.

## Data Loading And Provenance

- `csv_loader.load_csv` detects Tektronix-style preambles, headers, units,
  delimiters, and numeric columns.
- `DataSession.from_path` records source path, size, modification time, and
  SHA-256 hash.
- `shot_metadata.ensure_sidecar` writes `<shot>.meta.json` with source hash,
  row/column metadata, scope model, detected sample interval, QC status, and
  human-fillable lab fields. It does not store waveform arrays.
- `data_quality.quality_report` checks row count, columns, time monotonicity,
  duplicate/backwards timestamps, large sample gaps, and non-finite values.

## Formula Sandbox

Channel formulas are evaluated by `signal_tools.evaluate_formula` using a
restricted AST allowlist. Imports, attributes, subscripting tricks, and unknown
names are rejected. Available helpers include:

- `baseline(y, t, end)`: subtracts the mean of samples with `t <= end`.
- `lowpass(y, t, cutoff_hz)`: zero-phase Butterworth low-pass filtering.
- `integrate(y, t)`: cumulative trapezoidal integration.
- `gradient(y, t)`: numerical derivative using NumPy gradient.
- `movmean(y, n)`: centered moving average.
- `samples(seconds)`: convert a duration to sample count from the active time
  base.

## Filtering

The low-pass helper uses SciPy Butterworth filtering with zero-phase
forward/backward application when SciPy is available. This is intended for
display/analysis overlays, not for reconstructing a causal instrument response.
Users should state the cutoff and whether filtering was enabled in any exported
figure or notebook result.

## Integration And Differentiation

- Integration uses the trapezoidal rule.
- Differentiation uses `numpy.gradient`, which uses central differences inside
  the record and one-sided differences at the edges.
- Closed-form regression tests cover sine/cosine integration and gradients,
  moving averages, baseline removal, and low-pass behavior.

## Plot Decimation

The main pyqtgraph view uses view-dependent downsampling after a curve has been
added to a ViewBox. Export paths and deterministic tools operate on source
arrays or explicit derived arrays, not on the screen-decimated trace.

For non-interactive paths, `csv_loader.minmax_decimate` preserves local minima
and maxima in each bin so narrow pulses remain visible in overview plots.

## Calibration

`calibration.fit_forced_origin_gain` fits a forced-origin relation:

```text
reference ≈ gain * source
gain = dot(source, reference) / dot(source, source)
```

It reports residual RMS and an approximate confidence interval. This is a
linear calibration check, not proof of sensor correctness outside the fitted
window.

## Saturation And RLC Reconstruction

`saturation_recovery.py` and `rlc_reconstruct.py` produce model overlays. They
are not replacement measurements.

The RLC reconstruction uses an overdamped discharge model:

```text
I(t) = A * (exp(-(t - t0) / tau_d) - exp(-(t - t0) / tau_r)), t > t0
```

Samples above a known saturation level are treated as censored lower bounds.
Clean samples contribute ordinary residuals; censored samples penalize only
model curves that fall below the censoring level. A second reference sensor can
contribute clean samples over a trusted time window.

Bootstrap uncertainty is deterministic in the current implementation
(`default_rng(0)`) so reports are reproducible. Reported intervals describe
model uncertainty under the chosen assumptions; they do not include all
instrument, calibration, or topology uncertainty.

The real-data P1 benchmark is:

```bash
python3 scripts/backtest_rlc_reconstruction.py
```

On the 6.6 kA `T0000.CSV` shot, the default benchmark fits `CH1` with the
`BBCM v2` preset, uses `CH2` as a Pearson reference from 0-5 ms, treats the
BBCM as censored above 6000 A, focuses on 5-40 ms, and verifies post-window
faithfulness through 150 ms.

## Local AI Boundary

The local LLM never computes waveform statistics directly. It sees visible
summaries, small previews, and optional paper excerpts. It may request fixed
actions through `chat_actions.run_tool`; unknown tools are rejected. Numeric
results come from deterministic Python modules and should be cited as tool
outputs, not model calculations.
