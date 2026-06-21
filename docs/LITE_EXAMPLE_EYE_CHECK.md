# Scope Analyzer Lite - Visual Example Check

Use this checklist after rebuilding or changing the Lite UI. It is meant for an
eye check: the automated benchmark proves the numerical tools run, while this
helps catch confusing UI, broken overlays, bad labels, clipped buttons, or plots
that no longer look scientifically sensible.

## How To Start

1. Launch `dist/ScopeAnalyzerLite.app`.
2. Use the top-toolbar `Examples` menu.
3. Load each numbered benchmark dataset.
4. Use the `Example loaded` panel's `Run suggested tool` button first.
5. Optionally run the listed tool from `Tools & libraries` manually to check
   that the explicit path still works.
6. Capture anything suspicious with a screenshot and the dataset number.

The original CSVs should remain read-only. Filters, formulas, smoothing,
reconstruction curves, and generated traces are display/session data unless you
explicitly export a new file.

## General Pass/Fail Checklist

- The selected example loads without an alert.
- The file chip shows the expected file name and a sensible row count.
- Channel cards appear with readable labels, preset, gain, offset, formula, and
  unit controls.
- Buttons and text stay inside their panels at normal window size.
- The plot is not blank and has visible axes, grid, legend, and channel colours.
- The selected tool opens a clear parameter panel rather than only a placeholder.
- The `Example loaded` panel names a sensible tool, column, parameters, and
  expected result before running anything.
- Running the tool produces either a readable report, a visible overlay, or both.
- Any model/reconstruction overlay is visually distinguishable from measured data.
- Export/copy options still produce a PNG/SVG/JPG-style output when used.

## Dataset-By-Dataset Eye Checks

| # | Dataset | Load From Examples Menu | What To Try | What Should Look Reasonable |
| --- | --- | --- | --- | --- |
| 01 | Clean RLC pulse | `01 - clean RL pulse` | Statistics, derivative, integral, RLC | A smooth current pulse with no saturation; RLC overlay should follow the trace closely. |
| 02 | BBCM clipped hidden peak | `02 - BBCM clipped hidden peak` | Saturation estimate, Recover hidden peak/RLC | Flat clipping around 6 kA and a clearly labeled reconstructed hidden peak above the clipped region. |
| 03 | Low-pass ringing | `03 - low-pass ringing` | Low-pass at 15 kHz, FFT | Noisy/ringing trace becomes smoother; FFT or tool report should point to high-frequency ringing. |
| 04 | FFT two-tone | `04 - FFT two-tone` | FFT / dominant frequency | The report should identify the dominant tone near the intended high-frequency component. |
| 05 | Calibration pair | `05 - calibration pair` | Formula builder, reference calibration | Converted sensor trace should overlap the reference after applying the fitted gain. |
| 06 | dI/dt voltage 166 uH | `06 - dI/dt voltage 166 uH` | Derivative / dI/dt | Derived `dI/dt` should explain the inductive-voltage channel using `V = L*dI/dt`. |
| 07 | Charge integral | `07 - charge integral` | Integral | Integrated charge should rise monotonically and finish near the known synthetic charge. |
| 08 | Moving-average noise | `08 - moving average noise` | Moving average | Smoothed plateau should reduce random noise without shifting the main pulse timing. |
| 09 | Spike anomalies | `09 - spikes anomalies` | Anomaly scan | Injected spikes should be called out without treating the whole pulse as bad. |
| 10 | Quality gap/NaN/duplicate | `10 - quality gap nan duplicate` | CSV quality report | The report should flag timing gaps/duplicates and nonfinite data. |
| 11 | Baseline offset | `11 - baseline offset` | Formula builder with baseline correction | Baseline-corrected trace should move the pre-trigger region close to zero. |
| 12 | Soft saturation | `12 - soft saturation` | Saturation estimate, RLC | Compression above the monitor limit should be recognized as a saturation-like measurement problem. |
| 13 | Four-module balance | `13 - module balance` | Statistics, anomaly scan | Four module currents should be comparable but not identical; imbalance should be visible in stats. |
| 14 | Negative pulse | `14 - negative pulse` | Statistics, RLC | Negative-polarity current should plot and analyze correctly without sign confusion. |
| 15 | V/I/dI-dt 166 uH | `15 - V/I/dI-dt 166 uH` | Derivative, FFT, statistics | Voltage, current, and dI/dt channels should tell a consistent RL-drive story. |

## What To Send Back If Something Looks Wrong

Please include:

- Dataset number and name.
- Tool you ran.
- Screenshot of the full app window.
- One sentence describing what looked wrong, for example: `button clipped`,
  `legend overlaps trace`, `RLC overlay misses clean data`, or `filter did not
  visibly change the noisy trace`.
