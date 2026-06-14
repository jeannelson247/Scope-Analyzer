"""
saturation_recovery.py - Estimate true current when a monitor saturates.

Physics context: a clipped current monitor (scope range or sensor limit)
shows a flat top, but the true waveform keeps evolving. For capacitor-bank
discharges the plateau droop is locally ~linear, so the unclipped segments
BEFORE and AFTER the clipped run carry enough information to reconstruct
the lost peak by extrapolation - and an unclipped reference channel
(e.g. a Pearson) cross-calibrated in a clean window gives an independent
second estimate.

This is a deterministic NumPy tool: the LLM may *call* it and *interpret*
its output, but every number here comes from regression, not generation.

Method:
  1. Clip detection: the longest run of samples >= (1 - clip_tol) * max.
  2. Linear fits (np.polyfit, cov=True) on the unclipped segments
     adjacent to the run (pre-clip rise excluded: only data after the
     pulse reached 60% of clip level; post-clip until 50% fall).
  3. Extrapolate both lines across the clipped interval; the estimate of
     the true plateau/peak is the post-fit value at clip start (droop
     slope projected backwards), with a 95% prediction band from the
     parameter covariance. The pre-fit is reported as a cross-check.
  4. Reference cross-calibration (optional): ratio y/y_ref over a clean
     calibration window (default 0-5 ms or the rising flank below 80% of
     clip level); estimated true peak = ratio * ref peak.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SaturationReport:
    label: str
    clipped: bool
    text: str
    # structured fit data for plot overlays:
    # {"clip": (x0, x1), "rise": {"a","b","x0","x1"},
    #  "droop": {"a","b","x0","x1"}, "intersection": (x, y)}
    overlay: dict | None = None


def _linfit_with_ci(x: np.ndarray, y: np.ndarray, x_eval: float):
    """Linear fit y = a x + b; returns (value at x_eval, 95% CI, slope,
    intercept)."""
    if x.size < 8:
        return None
    (a, b), cov = np.polyfit(x, y, 1, cov=True)
    v = np.array([x_eval, 1.0])
    var = float(v @ cov @ v)
    return a * x_eval + b, 1.96 * np.sqrt(max(var, 0.0)), a, b


def estimate_true_current(x: np.ndarray, y: np.ndarray, label: str = "",
                          y_ref: np.ndarray | None = None,
                          ref_label: str = "",
                          clip_tol: float = 0.02,
                          min_clip_run: int = 200,
                          cal_window: tuple[float, float] | None = None,
                          sat_level: float | None = None
                          ) -> SaturationReport:
    """x in display units (typically ms), y the (possibly clipped) channel,
    y_ref an optional unclipped reference on the same x grid.

    sat_level: KNOWN saturation threshold in display units (e.g. a busbar
    monitor that compresses above 1500 A/module -> 6000 A summed). When
    given, everything above it is treated as invalid regardless of shape -
    this is the correct mode for SOFT saturation, where the reading keeps
    varying and the flatness detector rightly finds no hard clip."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ymax = float(np.nanmax(y))
    lines: list[str] = []

    # ---- 1. clipped run ---------------------------------------------------
    if sat_level is not None:
        near = y >= sat_level
    else:
        near = y >= (1.0 - clip_tol) * ymax
    runs, s = [], None
    for i, flag in enumerate(near):
        if flag and s is None:
            s = i
        elif not flag and s is not None:
            runs.append((s, i)); s = None
    if s is not None:
        runs.append((s, len(near)))
    if sat_level is not None:
        # explicit threshold: no shape tests needed, just a minimum run
        runs = [(a, b) for a, b in runs if b - a >= min_clip_run]
        if not runs:
            return SaturationReport(
                label, False,
                f"{label}: no run above the given saturation level "
                f"{sat_level:g} (signal max {ymax:.4g}).")
        # noise makes the reading straddle the level near entry/exit and
        # splits the run; data between the FIRST and LAST crossing is all
        # suspect (still compressed), so use the hull of all runs - and
        # extend it through the guard band just below the level, where a
        # softly-saturating sensor still reads compressed values
        guard = 0.02 * abs(sat_level)
        a0, b0 = runs[0][0], runs[-1][1]
        while a0 > 0 and y[a0 - 1] > sat_level - guard:
            a0 -= 1
        while b0 < len(y) and y[b0] > sat_level - guard:
            b0 += 1
        runs = [(a0, b0)]
    else:
        # a clipped run must also be LONG relative to the pulse itself -
        # otherwise the naturally rounded crest of an unclipped pulse (a
        # few hundred near-max samples) is mistaken for saturation
        pulse_samples = int(np.count_nonzero(y >= 0.5 * ymax))
        min_run = max(min_clip_run, int(0.08 * pulse_samples))
        runs = [(a, b) for a, b in runs if b - a >= min_run]
        # flatness filter: true clipping pins the (smoothed) signal to a
        # constant; a slow droop merely PASSES THROUGH the near-max band
        # and shows a trend spanning most of it. Reject trending runs.
        flat_runs = []
        for ra, rb in runs:
            seg = y[ra:rb]
            w = max(11, min(501, (rb - ra) // 10) | 1)
            k = np.ones(w) / w
            sm = np.convolve(seg, k, mode="valid")
            if sm.size and (np.nanmax(sm) - np.nanmin(sm)) \
                    < 0.5 * clip_tol * ymax:
                flat_runs.append((ra, rb))
        runs = flat_runs
        if not runs:
            return SaturationReport(
                label, False,
                f"{label}: no saturated run detected "
                f"(no >= {min_clip_run}-sample stretch within "
                f"{100*clip_tol:.0f}% of max {ymax:.4g}). If this monitor "
                f"saturates SOFTLY at a known level, re-run with "
                f"sat_level set (e.g. 'estimate saturation with sat "
                f"level 6000').")
    a, b = max(runs, key=lambda r: r[1] - r[0])
    t0c, t1c = x[a], x[b - 1]
    lines.append(
        f"{label}: saturated at ~{ymax:.4g} from {t0c:.3g} to {t1c:.3g} "
        f"({t1c - t0c:.3g} units, {b - a:,} samples).")

    # ---- 2. adjacent unclipped fits ---------------------------------------
    # post-clip: from clip end until y falls below 50% of clip level...
    yb = y[b:]
    below = np.flatnonzero(yb < 0.5 * ymax)
    post_end = b + (below[0] if below.size else yb.size)
    # ...but only the EARLY part of that segment: the late part may contain
    # the pulse's fast fall, which would corrupt the droop slope. Keep the
    # first post_frac (35%) closest to the clip, with a sane minimum.
    post_frac = 0.35
    post_end = b + max(min_clip_run // 2,
                       int(post_frac * (post_end - b))) \
        if post_end > b else post_end
    post_end = min(post_end, len(y))
    fit_post = _linfit_with_ci(x[b:post_end], y[b:post_end], float(t0c))
    # pre-clip: from 60% of clip level up to clip start (top of the rise)
    pre_start_candidates = np.flatnonzero(y[:a] >= 0.6 * ymax)
    fit_pre = None
    if pre_start_candidates.size:
        p0 = pre_start_candidates[0]
        fit_pre = _linfit_with_ci(x[p0:a], y[p0:a], float(t0c))

    overlay: dict = {"clip": (float(t0c), float(t1c))}
    est, ci = None, None
    if fit_post is not None:
        est, ci, slope, b_post = fit_post
        overlay["droop"] = {"a": float(slope), "b": float(b_post),
                            "x0": float(t0c),
                            "x1": float(x[min(post_end - 1, len(x) - 1)])}
        lines.append(
            f"  droop-slope estimate (post-saturation fit, "
            f"{post_end - b:,} samples, slope {slope:+.4g}/unit projected "
            f"back to {t0c:.3g}): true peak ~ {est:.5g} +/- {ci:.3g} (95%).")
    if fit_pre is not None:
        v_pre, ci_pre, slope_pre, b_pre = fit_pre
        overlay["rise"] = {"a": float(slope_pre), "b": float(b_pre),
                           "x0": float(x[p0]), "x1": float(t0c)}
        lines.append(
            f"  rise-side cross-check (pre-saturation fit): {v_pre:.5g} "
            f"+/- {ci_pre:.3g}.")

    # two-line intersection (rise fit x droop fit): where the projected
    # rise meets the projected droop - the reconstructed peak corner
    if fit_post is not None and fit_pre is not None \
            and abs(slope_pre - slope) > 1e-12:
        t_x = (b_post - b_pre) / (slope_pre - slope)
        i_x = slope_pre * t_x + b_pre
        if t0c - 0.5 * (t1c - t0c) <= t_x <= t1c + 0.5 * (t1c - t0c):
            overlay["intersection"] = (float(t_x), float(i_x))
            overlay["rise"]["x1"] = float(t_x)
            overlay["droop"]["x0"] = float(t_x)
            lines.append(
                f"  two-slope intersection: rise and droop projections "
                f"meet at t = {t_x:.3g}, I = {i_x:.5g} - reconstructed "
                f"peak corner estimate.")
        else:
            lines.append(
                f"  two-slope intersection falls at t = {t_x:.3g}, "
                f"outside the saturated window - rise fit is probably "
                f"too short/curved to extrapolate; trust the droop "
                f"estimate above.")

    # ---- 3. reference cross-calibration ----------------------------------
    if y_ref is not None:
        y_ref = np.asarray(y_ref, dtype=np.float64)
        if cal_window is not None:
            mcal = (x >= cal_window[0]) & (x <= cal_window[1])
        else:
            mcal = np.zeros(len(y), dtype=bool)
            mcal[:a] = True                       # rising flank only
        # clean part: both channels meaningfully nonzero, y safely below
        # the (known or apparent) saturation level
        lvl = sat_level if sat_level is not None else ymax
        mcal &= (y < 0.8 * lvl) & (np.abs(y_ref) > 0.05 * np.nanmax(
            np.abs(y_ref))) & (np.abs(y) > 0.05 * lvl)
        if mcal.sum() >= 16:
            r = y[mcal] / y_ref[mcal]
            ratio, rsd = float(np.nanmedian(r)), float(np.nanstd(r))
            ref_pk = float(np.nanmax(np.abs(y_ref)))
            lines.append(
                f"  cross-calibration vs {ref_label or 'reference'}: "
                f"ratio {ratio:.4g} +/- {rsd:.2g} over {int(mcal.sum()):,} "
                f"clean samples -> estimated true peak "
                f"{ratio * ref_pk:.5g}.")
            if est is not None:
                agree = 100 * abs(ratio * ref_pk - est) / max(abs(est), 1e-12)
                lines.append(
                    f"  the two methods differ by {agree:.1f}% - "
                    + ("good agreement." if agree < 5 else
                       "check calibration window / linearity."))
        else:
            lines.append("  cross-calibration skipped: not enough clean "
                         "overlapping samples in the calibration window.")

    lines.append("  note: linear extrapolation assumes the droop stays "
                 "linear inside the saturated window - treat as an "
                 "estimate, not a measurement.")
    if sat_level is None:
        lines.append(
            "  note: run detected by SHAPE. If this monitor saturates "
            "SOFTLY at a known level (reading compressed but still "
            "varying above it), the fits above may include corrupted "
            "samples - re-run with sat_level (e.g. 'estimate saturation "
            "with sat level 6000') to fit only trustworthy data.")
    return SaturationReport(label, True, "\n".join(lines), overlay=overlay)
