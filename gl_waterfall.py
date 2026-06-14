"""
gl_waterfall.py - GPU-accelerated 3D rendering for large scope datasets.

Why this exists: matplotlib's Axes3D is software-rendered and immediate-
mode - fine for a 150x150 surface, hopeless for 1.25M-point waveforms.
This module uses pyqtgraph.opengl (PyOpenGL), which uploads float32
vertex buffers to the GPU once and lets Metal/OpenGL redraw them during
rotation. On an M4 Pro this renders multi-million-point cascades at
interactive frame rates.

Optimization strategy (fastest available, in order):
  1. GPU retained mode  - vertices live in VRAM; rotation costs ~0 CPU.
  2. float32 buffers    - half the memory bandwidth of float64.
  3. min-max decimation - a 1.25M-point trace carries at most ~2x the
     horizontal pixel count of visually distinct information; we keep
     peaks (csv_loader.minmax_decimate) and cap each trace at
     MAX_PTS_PER_TRACE vertices. Spikes survive; bandwidth drops 10x.
  4. antialias off for > 100k vertices (GPU fill-rate, not vertex count,
     is the bottleneck on retina displays).

Graceful degradation: if PyOpenGL is missing, WATERFALL_AVAILABLE is
False and the caller should fall back to the matplotlib tab.
    pip install PyOpenGL PyOpenGL-accelerate
"""
from __future__ import annotations

import numpy as np

try:
    import pyqtgraph.opengl as gl
    from pyqtgraph import mkColor
    WATERFALL_AVAILABLE = True
except Exception:                                  # PyOpenGL not installed
    gl = None
    WATERFALL_AVAILABLE = False

MAX_PTS_PER_TRACE = 200_000
MAX_SURFACE_CELLS = 600 * 600


def _decimate(x: np.ndarray, y: np.ndarray, target: int):
    from csv_loader import minmax_decimate
    return minmax_decimate(np.asarray(x), np.asarray(y), target)


class GLWaterfall:
    """Thin wrapper building a GLViewWidget with line cascades and
    surfaces from large arrays. Owns no Qt window - embed its `widget`."""

    def __init__(self):
        if not WATERFALL_AVAILABLE:
            raise ImportError(
                "pyqtgraph.opengl unavailable - pip install PyOpenGL "
                "PyOpenGL-accelerate")
        self.widget = gl.GLViewWidget()
        self.widget.setCameraPosition(distance=2.6, elevation=24,
                                      azimuth=-60)
        self._items: list = []
        grid = gl.GLGridItem()
        grid.setSize(2, 2, 1)
        grid.setSpacing(0.1, 0.1, 0.1)
        self.widget.addItem(grid)

    # ------------------------------------------------------------------
    def clear(self):
        for it in self._items:
            self.widget.removeItem(it)
        self._items = []

    @staticmethod
    def _norm(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
        span = hi - lo
        return (v - lo) / (span if span else 1.0)

    def add_cascade(self, t: np.ndarray, traces: list[tuple[str,
                    np.ndarray]], colors: list | None = None):
        """traces: [(name, y), ...] all on grid t. Renders each trace as
        a GPU line at depth k, normalized into a unit box. Returns the
        (t_lo, t_hi, y_lo, y_hi) mapping for axis annotation."""
        self.clear()
        t = np.asarray(t, dtype=np.float64)
        y_lo = min(float(np.nanmin(y)) for _n, y in traces)
        y_hi = max(float(np.nanmax(y)) for _n, y in traces)
        t_lo, t_hi = float(t[0]), float(t[-1])
        n = len(traces)
        for k, (name, y) in enumerate(traces):
            xs, ys = _decimate(t, np.asarray(y, dtype=np.float64),
                               MAX_PTS_PER_TRACE)
            pts = np.empty((len(xs), 3), dtype=np.float32)
            pts[:, 0] = 2 * self._norm(xs, t_lo, t_hi) - 1        # x
            pts[:, 1] = (2 * k / max(n - 1, 1) - 1) * 0.8         # depth
            pts[:, 2] = self._norm(ys, y_lo, y_hi)                # height
            if colors and k < len(colors):
                c = mkColor(colors[k])
                rgba = (c.redF(), c.greenF(), c.blueF(), 0.95)
            else:
                frac = k / max(n - 1, 1)
                rgba = (0.1 + 0.8 * frac, 0.35, 0.95 - 0.8 * frac, 0.95)
            item = gl.GLLinePlotItem(
                pos=pts, color=rgba, width=1.5,
                antialias=len(xs) <= 100_000, mode="line_strip")
            self.widget.addItem(item)
            self._items.append(item)
        return t_lo, t_hi, y_lo, y_hi

    def add_surface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    cmap_name: str = "viridis"):
        """x (nx,), y (ny,), z (ny, nx) - GPU surface with per-vertex
        colormap. Large grids are mean-binned to MAX_SURFACE_CELLS."""
        self.clear()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        # bin down huge grids (GPU happily renders 600x600 = 720k tris)
        while z.shape[0] * z.shape[1] > MAX_SURFACE_CELLS:
            if z.shape[1] >= z.shape[0]:
                m = z.shape[1] // 2 * 2
                z = 0.5 * (z[:, 0:m:2] + z[:, 1:m:2])
                x = 0.5 * (x[0:m:2] + x[1:m:2])
            else:
                m = z.shape[0] // 2 * 2
                z = 0.5 * (z[0:m:2, :] + z[1:m:2, :])
                y = 0.5 * (y[0:m:2] + y[1:m:2])
        from matplotlib import cm
        zn = self._norm(z, float(np.nanmin(z)), float(np.nanmax(z)))
        colors = cm.get_cmap(cmap_name)(zn)
        colors = colors.astype(np.float32)
        surf = gl.GLSurfacePlotItem(
            x=(2 * self._norm(x, x.min(), x.max()) - 1).astype(np.float32),
            y=(2 * self._norm(y, y.min(), y.max()) - 1).astype(np.float32),
            z=zn.T.astype(np.float32),          # GLSurface expects (nx, ny)
            colors=np.transpose(colors, (1, 0, 2)).reshape(-1, 4),
            shader="shaded", smooth=False)
        self.widget.addItem(surf)
        self._items.append(surf)
