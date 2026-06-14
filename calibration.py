"""
calibration.py — Reference-channel gain calibration (forced-through-origin).

Pure NumPy/dataclass implementation of the "Reference calibration" fit used
by the desktop UI's gcal panel. Pulling this out of app.py means:

  * the fit math can be unit-tested and reused (CLI tools, PulseLab port,
    notebooks) without importing Qt at all
  * the UI layer (_fit_reference_gain in app.py) is reduced to: gather
    arrays -> call fit_forced_origin_gain -> format the result text

This mirrors the MATLAB ratio used across the lab's busbar/Pearson/
Rogowski calibration scripts:

    ratio = dot(source, reference) / dot(source, source)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class CalibrationError(ValueError):
    """Raised when a reference-calibration fit cannot be computed."""


@dataclass
class GainFitResult:
    slope: float
    r2: float
    ci_lo: float
    ci_hi: float
    n_samples: int
    window: tuple[float, float]


def fit_forced_origin_gain(x: np.ndarray, y_src: np.ndarray, y_ref: np.ndarray,
                            lo: float, hi: float,
                            ci_z: float = 1.645) -> GainFitResult:
    """Fit ``slope = dot(src, ref) / dot(src, src)`` over ``x in [lo, hi]``.

    Parameters
    ----------
    x : the shared X axis (seconds or ms, whatever the caller is using)
    y_src : source-channel samples (the channel whose gain gets multiplied)
    y_ref : reference-channel samples
    lo, hi : fit window bounds along ``x``
    ci_z : z-score for the approximate confidence interval (default 1.645
        -> ~90% CI, matching the existing UI text)

    Returns
    -------
    GainFitResult with the fitted slope, R^2, approximate CI, sample count,
    and the (lo, hi) window actually used.

    Raises
    ------
    CalibrationError if the window contains fewer than 3 finite samples or
    the source channel is effectively zero in that window.
    """
    x = np.asarray(x, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y_src) & np.isfinite(y_ref)
    mask &= (x >= lo) & (x <= hi)
    xs = y_src[mask]
    yr = y_ref[mask]
    if xs.size < 3:
        raise CalibrationError("Not enough samples inside the fit window.")

    denom = float(np.dot(xs, xs))
    if abs(denom) < 1e-18:
        raise CalibrationError("Source channel is effectively zero there.")

    slope = float(np.dot(xs, yr) / denom)
    pred = slope * xs
    rss = float(np.sum((yr - pred) ** 2))
    tss = float(np.sum((yr - np.mean(yr)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    se = math.sqrt(max(rss, 0.0) / max(xs.size - 1, 1) / denom)

    return GainFitResult(
        slope=slope,
        r2=r2,
        ci_lo=slope - ci_z * se,
        ci_hi=slope + ci_z * se,
        n_samples=int(xs.size),
        window=(float(lo), float(hi)),
    )


def format_gain_fit(result: GainFitResult, channel_label: str,
                     axis_label: str) -> str:
    """Render a GainFitResult as the multi-line status text shown in the UI."""
    return (
        f"Applied gain multiplier to {channel_label}.\n"
        f"slope = {result.slope:.6g}\n"
        f"approx. 90% CI = [{result.ci_lo:.6g}, {result.ci_hi:.6g}]\n"
        f"R^2 = {result.r2:.6g}\n"
        f"window = {result.window[0]:.4g} to {result.window[1]:.4g} ({axis_label})\n"
        f"samples = {result.n_samples:,}"
    )
