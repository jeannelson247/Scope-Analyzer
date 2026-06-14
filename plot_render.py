"""
plot_render.py — Shared large-data decimation and curve styling.

This is the single place that decides how raw (potentially multi-million
point) channel arrays become pyqtgraph curves. Centralizing it gives:

  * one tunable performance knob (LARGE_TRACE_THRESHOLD /
    RENDER_TARGET_POINTS) instead of scattered magic numbers
  * peak-preserving pre-decimation for very large traces, on top of
    pyqtgraph's own auto-downsampling, so the *first* render of a
    multi-million-sample CSV is fast, not just subsequent redraws
  * one styling function (`make_curve`) so the 2D scope view, overlay
    shots, and any future modes (PulseLab parity) all look and behave
    the same

The deterministic decimation itself (`minmax_decimate`) lives in
csv_loader.py and is unchanged; this module just decides *when* to call it.
"""
from __future__ import annotations

import numpy as np

from csv_loader import minmax_decimate

# NOTE: pyqtgraph (and therefore Qt) is imported lazily inside make_curve,
# not at module top. That keeps the pure-NumPy decimation logic
# (decimate_for_render + the threshold constants) importable and unit-
# testable in a headless environment (CI, notebooks) with no Qt present.

# Traces with more samples than this get a peak-preserving min/max
# pre-decimation pass before being handed to pyqtgraph. pyqtgraph's
# setDownsampling(auto=True) still runs on top of this for pan/zoom, but
# pre-decimation keeps the *initial* setData() call - and memory use -
# bounded for multi-million-point oscilloscope captures.
LARGE_TRACE_THRESHOLD = 200_000

# Target point count after pre-decimation. Comfortably above typical
# screen resolution so peaks/spikes stay visible while still being a
# large reduction from multi-million-sample inputs.
RENDER_TARGET_POINTS = 100_000


def decimate_for_render(x: np.ndarray, y: np.ndarray,
                         threshold: int = LARGE_TRACE_THRESHOLD,
                         target: int = RENDER_TARGET_POINTS):
    """Return ``(x, y)``, peak-preserving-decimated if larger than ``threshold``.

    Below the threshold the inputs are returned unchanged (no copy), so
    interactive zoom/pan on typical scope captures stays exact. This is a
    *display* decision only - exported/analyzed data is untouched.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size <= threshold:
        return x, y
    return minmax_decimate(x, y, target)


def make_curve(x: np.ndarray, y: np.ndarray, color: str, name: str,
               width: float = 1.4, decimate: bool = True):
    """Build a styled ``pyqtgraph.PlotDataItem`` (pen, name, finite-skip).

    Optionally pre-decimates large traces (see :func:`decimate_for_render`).

    IMPORTANT: this does NOT enable auto peak-downsampling or clip-to-view.
    Those are view-dependent and must be set *after* the curve is added to
    a ViewBox -- calling them on a detached item can clip the whole trace
    to nothing and break autorange (a blank plot with runaway axes). Call
    :func:`enable_view_downsampling` right after ``addItem``.

    pyqtgraph is imported here (not at module top) so importing this module
    for its decimation logic does not require Qt.
    """
    import pyqtgraph as pg
    if decimate:
        x, y = decimate_for_render(x, y)
    return pg.PlotDataItem(x, y, pen=pg.mkPen(color, width=width),
                           name=name, skipFiniteCheck=True)


def enable_view_downsampling(curve) -> None:
    """Enable auto peak-downsampling + clip-to-view on a curve.

    Call this AFTER the curve has been added to a ViewBox/PlotItem.
    pyqtgraph resolves both against the item's parent view, so setting
    them before the item has a view hides the trace and breaks autorange.
    """
    curve.setDownsampling(auto=True, method="peak")
    curve.setClipToView(True)
