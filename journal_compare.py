"""
journal_compare.py - deterministic journal-figure comparison helpers.

This is intentionally lightweight. It gives the assistant a safe vocabulary
for comparing Scope Studio output with extracted journal figures without
claiming physical equivalence from image similarity alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StyleSpec:
    font_size_pt: float
    line_width_pt: float
    aspect_ratio: float
    has_legend: bool
    palette_name: str = ""


@dataclass(frozen=True)
class StyleComparison:
    score: float
    report: str


def compare_style(candidate: StyleSpec, reference: StyleSpec) -> StyleComparison:
    penalties: list[tuple[float, str]] = []
    if reference.font_size_pt > 0:
        rel = abs(candidate.font_size_pt - reference.font_size_pt) \
            / reference.font_size_pt
        if rel > 0.12:
            penalties.append((min(rel, 1.0), "font size differs"))
    if reference.line_width_pt > 0:
        rel = abs(candidate.line_width_pt - reference.line_width_pt) \
            / reference.line_width_pt
        if rel > 0.20:
            penalties.append((min(rel, 1.0), "line width differs"))
    if math.isfinite(reference.aspect_ratio) and reference.aspect_ratio > 0:
        rel = abs(candidate.aspect_ratio - reference.aspect_ratio) \
            / reference.aspect_ratio
        if rel > 0.10:
            penalties.append((min(rel, 1.0), "aspect ratio differs"))
    if candidate.has_legend != reference.has_legend:
        penalties.append((0.15, "legend presence differs"))
    penalty = min(sum(p for p, _ in penalties), 1.0)
    score = 100.0 * (1.0 - penalty)
    if penalties:
        notes = "; ".join(note for _, note in penalties)
    else:
        notes = "style matches the supplied reference within tolerances"
    return StyleComparison(score=score, report=f"Style score {score:.0f}/100: {notes}.")


def compare_digitized_curves(x_ref, y_ref, x_candidate, y_candidate) -> str:
    """Compare already-digitized curves using interpolation.

    Axis calibration/digitization must happen outside this helper, either
    manually or with a reviewed future tool.
    """
    import numpy as np

    xr = np.asarray(x_ref, dtype=float)
    yr = np.asarray(y_ref, dtype=float)
    xc = np.asarray(x_candidate, dtype=float)
    yc = np.asarray(y_candidate, dtype=float)
    m = np.isfinite(xr) & np.isfinite(yr)
    xr, yr = xr[m], yr[m]
    m = np.isfinite(xc) & np.isfinite(yc)
    xc, yc = xc[m], yc[m]
    if xr.size < 4 or xc.size < 4:
        return "Curve comparison: not enough digitized points."
    lo, hi = max(np.min(xr), np.min(xc)), min(np.max(xr), np.max(xc))
    if hi <= lo:
        return "Curve comparison: no overlapping x-range."
    grid = np.linspace(lo, hi, 400)
    yr_i = np.interp(grid, xr, yr)
    yc_i = np.interp(grid, xc, yc)
    scale = max(float(np.nanmax(yr_i) - np.nanmin(yr_i)), 1e-12)
    rmse = float(np.sqrt(np.nanmean((yc_i - yr_i) ** 2)))
    return (
        f"Curve comparison over {lo:.4g}..{hi:.4g}: "
        f"RMSE={rmse:.4g} ({100 * rmse / scale:.2f}% of reference span). "
        "This checks plotted-shape similarity only, not experimental proof."
    )
