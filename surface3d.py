"""
surface3d.py - 3D surface viewer window for Scope Studio.

Self-contained Qt window (no changes to the 2D pipeline) that renders
3D surfaces with gradient colormaps using matplotlib's Axes3D embedded
in Qt - zero new dependencies (matplotlib is already required).

Features:
  * Demo surfaces: 3D Gaussian and Mexican-hat (sombrero) potential -
    useful as teaching examples and for verifying the renderer.
  * CSV loading, two layouts:
      - long format: columns x, y, z  (gridded via binning)
      - matrix format: first column = y values, header row = x values
  * Colormap selection (viridis/plasma/coolwarm/... - colorblind-safe
    defaults first).
  * 4th dimension: the surface can be colored by a separate scalar
    array C(x, y) instead of by height z - "embedding higher dimensions"
    as color (e.g. z = |B| while color = temperature).
  * Mouse: drag to rotate, scroll to zoom (matplotlib defaults).
  * Export PNG/SVG at print resolution.

The window is opened from Scope Studio via a View-menu action but can
also run standalone:  python3 surface3d.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import cm
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog,
                               QHBoxLayout, QHeaderView, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

COLORMAPS = ["viridis", "plasma", "cividis", "coolwarm",
             "RdBu_r", "magma", "turbo"]


# --------------------------------------------------------------------------
# Demo / analytic surfaces
# --------------------------------------------------------------------------
def gaussian_surface(n: int = 121, sigma: float = 1.0):
    """z = exp(-(x^2+y^2)/2sigma^2) on [-3sigma, 3sigma]^2."""
    ax = np.linspace(-3 * sigma, 3 * sigma, n)
    x, y = np.meshgrid(ax, ax)
    z = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return x, y, z


def mexican_hat_surface(n: int = 161, a: float = 1.0, b: float = 1.0):
    """Sombrero potential V(r) = -a*r^2 + b*r^4 (the 'Mexican hat' used
    for spontaneous symmetry breaking illustrations)."""
    lim = 1.6 * np.sqrt(a / (2 * b)) + 0.8
    ax = np.linspace(-lim, lim, n)
    x, y = np.meshgrid(ax, ax)
    r2 = x ** 2 + y ** 2
    z = -a * r2 + b * r2 ** 2
    return x, y, z


def load_table(path: str):
    """Robust tabular load: uses Scope Studio's csv_loader (handles
    instrument preambles, odd delimiters, big files) with a plain
    pandas fallback for ordinary CSVs."""
    try:
        from csv_loader import load_csv as _scope_load
        return _scope_load(path).df
    except Exception:
        import pandas as pd
        return pd.read_csv(path)


MAX_BINS = 150     # axes with more unique values are auto-binned


def _maybe_bin(df, col: str, max_bins: int = MAX_BINS):
    """Bin a high-cardinality axis to bin midpoints so ANY data size
    grids; z is then the mean per cell (built-in decimation)."""
    if df[col].nunique() <= max_bins:
        return df, col
    import pandas as pd
    name = f"{col} (binned)"
    df = df.copy()
    df[name] = pd.cut(df[col].astype(float), bins=max_bins) \
        .map(lambda iv: iv.mid).astype(float)
    return df, name


def pivot_grid(df, xc: str, yc: str, zc: str, cc: str | None = None):
    """Grid long-format data with user-chosen columns -> (x, y, z, c).
    High-cardinality axes are binned automatically (mean-aggregated),
    so file size is not a limit."""
    df, xc = _maybe_bin(df, xc)
    df, yc = _maybe_bin(df, yc)
    nx, ny = df[xc].nunique(), df[yc].nunique()
    if nx < 2 or ny < 2:
        raise ValueError(
            f"'{xc}' x '{yc}' do not form a grid ({nx} x {ny} unique "
            f"values). Time-series scope files are not surfaces - use "
            f"the 'Shot data 3D' tab for those.")
    piv = df.pivot_table(index=yc, columns=xc, values=zc, aggfunc="mean")
    # a real surface populates (nearly) every (x, y) cell; a time series
    # only covers a thin curve through the grid and renders as garbage
    fill = float(piv.notna().to_numpy().mean())
    if fill < 0.5:
        raise ValueError(
            f"'{xc}' x '{yc}' fills only {100*fill:.1f}% of a "
            f"{piv.shape[1]} x {piv.shape[0]} grid - this looks like a "
            f"time series, not a surface. Use the 'Shot data 3D' tab "
            f"for waveforms.")
    x, y = np.meshgrid(piv.columns.to_numpy(float),
                       piv.index.to_numpy(float))
    z = piv.to_numpy(float)
    c = None
    if cc:
        pc = df.pivot_table(index=yc, columns=xc, values=cc,
                            aggfunc="mean")
        c = pc.to_numpy(float)
    return x, y, z, c


def grid_from_csv(path: str):
    """Auto path: x/y/z(/c) columns if present, else matrix format."""
    df = load_table(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    if {"x", "y", "z"} <= set(cols):
        return pivot_grid(df, cols["x"], cols["y"], cols["z"],
                          cols.get("c"))
    import pandas as pd
    raw = pd.read_csv(path, index_col=0)
    x, y = np.meshgrid(raw.columns.to_numpy(float),
                       raw.index.to_numpy(float))
    return x, y, raw.to_numpy(float), None


def looks_like_time_series(df) -> bool:
    cols = [str(c).strip().lower() for c in df.columns]
    has_time = any(c in ("t", "time", "time_s", "time (s)", "seconds")
                   or "time" in c for c in cols[:2])
    numeric_cols = 0
    for col in df.columns:
        try:
            np.asarray(df[col].dropna().head(8), dtype=float)
            numeric_cols += 1
        except Exception:
            pass
    return has_time and numeric_cols >= 2


# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------
class Surface3DWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scope Studio - 3D surface view")
        self.resize(900, 700)
        # default demo: the Mexican-hat potential (requested example)
        self._data = mexican_hat_surface()
        self._color = None          # optional 4th-dimension array
        self._axis_labels = ("x", "y", "z")

        central = QWidget()
        v = QVBoxLayout(central)

        row = QHBoxLayout()
        self.cmb_demo = QComboBox()
        self.cmb_demo.addItems([
            "Mexican hat potential",
            "3D Gaussian",
            "Synthetic current surface",
            "Synthetic di/dt surface",
            "Synthetic inductive voltage surface",
        ])
        self.cmb_demo.activated.connect(self._load_demo)
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(COLORMAPS)
        self.cmb_cmap.activated.connect(lambda _i: self.redraw())
        btn_csv = QPushButton("Load CSV…")
        btn_csv.clicked.connect(self._load_csv)
        btn_export = QPushButton("Export…")
        btn_export.clicked.connect(self._export)
        row.addWidget(QLabel("Demo:"))
        row.addWidget(self.cmb_demo)
        row.addWidget(QLabel("Colormap:"))
        row.addWidget(self.cmb_cmap)
        row.addWidget(btn_csv)
        row.addWidget(btn_export)
        row.addStretch(1)
        v.addLayout(row)

        self.lbl_info = QLabel(
            "Drag to rotate - scroll to zoom. CSV formats: columns x,y,z "
            "(+ optional c for color-encoded 4th dimension) or a matrix "
            "with x in the header row and y in the first column.")
        self.lbl_info.setWordWrap(True)
        v.addWidget(self.lbl_info)

        self.fig = Figure(figsize=(7, 6), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        v.addWidget(self.canvas, 1)

        # browser-style tabs: analytic surfaces | shot data in 3D |
        # GPU-accelerated cascade for very large datasets
        self.tabs = QTabWidget()
        self.tabs.addTab(central, "Surfaces")
        self.tabs.addTab(self._build_shot_tab(), "Shot data 3D")
        self.tabs.addTab(self._build_gpu_tab(), "GPU 3D (fast)")
        self.tabs.addTab(self._build_vi_tab(), "V-I map")
        self.tabs.addTab(self._build_detail_tab(), "Detail + FFT")
        self.setCentralWidget(self.tabs)
        self._embedded_tab_offset = 0
        self.redraw()

    def _set_tab_index(self, index: int):
        self.tabs.setCurrentIndex(index + int(getattr(
            self, "_embedded_tab_offset", 0)))

    # ------------------------------------------------------------------
    # V-I trajectory map: the switching locus. Spikes that hide in time
    # series stand out as excursions from the operating path.
    def _build_vi_tab(self) -> QWidget:
        import pyqtgraph as pg
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        self.cmb_vi_x = QComboBox()
        self.cmb_vi_y = QComboBox()
        btn = QPushButton("Map visible window")
        btn.setToolTip(
            "Plots Y-channel vs X-channel over the MAIN plot's visible "
            "time range, colored early->late (viridis). Zoom the main "
            "plot first to control granularity; zoom here freely - up "
            "to 400k full-resolution points are used.")
        btn.clicked.connect(self._vi_map)
        row.addWidget(QLabel("X:"))
        row.addWidget(self.cmb_vi_x, 1)
        row.addWidget(QLabel("Y:"))
        row.addWidget(self.cmb_vi_y, 1)
        row.addWidget(btn)
        v.addLayout(row)
        self.lbl_vi = QLabel(
            "Pick channels (e.g. X = current, Y = IGBT voltage) and map "
            "the visible window. Color encodes time: dark = early, "
            "bright = late.")
        self.lbl_vi.setWordWrap(True)
        v.addWidget(self.lbl_vi)
        self.vi_plot = pg.PlotWidget()
        self.vi_plot.showGrid(x=True, y=True, alpha=0.2)
        v.addWidget(self.vi_plot, 1)
        return w

    def _vi_map(self):
        import pyqtgraph as pg
        p = self.parent()
        chans = getattr(p, "channels", []) or []
        labels = [c.display_label() for c in chans]
        nx = self.cmb_vi_x.currentText()
        ny = self.cmb_vi_y.currentText()
        ch_x = next((c for c in chans if c.display_label() == nx), None)
        ch_y = next((c for c in chans if c.display_label() == ny), None)
        if ch_x is None or ch_y is None:
            QMessageBox.information(self, "V-I map",
                                    "Pick X and Y channels (Refresh "
                                    "columns first).")
            return
        t = p._x()
        (x0, x1) = p.pi.vb.viewRange()[0]
        m = (t >= x0) & (t <= x1)
        try:
            xi = p._channel_data(ch_x)[m]
            yi = p._channel_data(ch_y)[m]
        except Exception as e:
            QMessageBox.warning(self, "V-I map", str(e))
            return
        step = max(1, xi.size // 400_000)
        xi, yi = xi[::step], yi[::step]
        self.vi_plot.clear()
        from matplotlib import cm
        cmap = cm.get_cmap("viridis")
        nseg = 200
        edges = np.linspace(0, xi.size, nseg + 1).astype(int)
        for k in range(nseg):
            a, b = edges[k], min(edges[k + 1] + 1, xi.size)
            if b - a < 2:
                continue
            r, g, bl, _ = cmap(k / (nseg - 1))
            self.vi_plot.plot(xi[a:b], yi[a:b],
                              pen=pg.mkPen(pg.mkColor(int(255 * r),
                                                      int(255 * g),
                                                      int(255 * bl),
                                                      200), width=1))
        self.vi_plot.setLabel("bottom", nx)
        self.vi_plot.setLabel("left", ny)
        self.lbl_vi.setText(
            f"{xi.size:,} points over {x0:.4g}..{x1:.4g} (main-plot "
            f"window). Color: dark = early, bright = late. Spikes show "
            f"as excursions leaving the main locus.")

    # ------------------------------------------------------------------
    # Detail + FFT: full-resolution snapshot of the visible window for
    # high-frequency ringing analysis (no decimation at all).
    def _build_detail_tab(self) -> QWidget:
        import pyqtgraph as pg
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        btn = QPushButton("Capture visible window (full res + FFT)")
        btn.setToolTip(
            "Snapshots the MAIN plot's visible time range at FULL sample "
            "resolution (no decimation) for every checked channel, and "
            "computes the amplitude spectrum (Hann window). Zoom into a "
            "switching edge first, then capture.")
        btn.clicked.connect(self._capture_detail)
        row.addWidget(btn)
        row.addStretch(1)
        v.addLayout(row)
        self.lbl_det = QLabel(
            "Zoom the main plot into a small window (e.g. one switching "
            "edge), then capture. Top: full-resolution traces. Bottom: "
            "spectrum - ringing appears as a sharp peak; the dominant "
            "frequency is annotated.")
        self.lbl_det.setWordWrap(True)
        v.addWidget(self.lbl_det)
        self.det_plot = pg.PlotWidget()
        self.det_plot.showGrid(x=True, y=True, alpha=0.2)
        v.addWidget(self.det_plot, 1)
        self.fft_plot = pg.PlotWidget()
        self.fft_plot.showGrid(x=True, y=True, alpha=0.2)
        self.fft_plot.setLogMode(False, True)
        self.fft_plot.setLabel("bottom", "frequency (kHz)")
        self.fft_plot.setLabel("left", "amplitude (dY/dt spectrum)")
        v.addWidget(self.fft_plot, 1)
        return w

    def _capture_detail(self):
        import pyqtgraph as pg
        p = self.parent()
        chans = self._checked_channels()
        if not chans:
            QMessageBox.information(self, "Detail",
                                    "Tick channels in Shot data 3D "
                                    "first (Refresh columns).")
            return
        t = p._x()
        raw_t = p._raw_x()
        (x0, x1) = p.pi.vb.viewRange()[0]
        m = (t >= x0) & (t <= x1)
        if m.sum() < 32:
            QMessageBox.information(self, "Detail",
                                    "Visible window has too few samples.")
            return
        dt = float(np.median(np.diff(raw_t[m])))      # seconds
        self.det_plot.clear()
        self.fft_plot.clear()
        notes = []
        for ch in chans:
            try:
                y = p._channel_data(ch)[m]
            except Exception:
                continue
            self.det_plot.plot(t[m], y, pen=pg.mkPen(ch.color, width=1),
                               name=ch.display_label())
            # ringing analysis on the derivative of the segment right
            # AFTER the strongest edge: differentiation flattens the
            # step (broadband impulse) while narrowband ringing keeps a
            # sharp peak, and the post-edge segment is where switching
            # ringing physically lives. Validated: 138 kHz and 35 kHz
            # bursts -> ~100x local prominence; step-only and pure-noise
            # controls -> ~3x. Gate at 8x.
            yd = np.diff(y)
            if yd.size < 64:
                continue
            k_edge = int(np.argmax(np.abs(yd)))
            seg = yd[k_edge + 3: k_edge + 3 + 4096]
            if seg.size < 64:
                seg = yd
            from signal_tools import amplitude_spectrum
            freq_hz, spec = amplitude_spectrum(seg, dt)   # tested pure core
            freq = freq_hz / 1e3                           # kHz
            if freq.size > 2:
                self.fft_plot.plot(freq[1:], np.maximum(spec[1:], 1e-12),
                                   pen=pg.mkPen(ch.color, width=1))
                hi = np.flatnonzero(freq > 1.0)
                if hi.size:
                    k = int(hi[int(np.argmax(spec[hi]))])
                    f_pk = float(freq[k])
                    nb = np.r_[spec[max(0, k - 150):max(0, k - 6)],
                               spec[k + 7:k + 150]]
                    prom = float(spec[k] / max(np.median(nb), 1e-12)) \
                        if nb.size else 0.0
                    if prom >= 8.0:
                        notes.append(
                            f"{ch.display_label()}: ringing at "
                            f"{f_pk:.1f} kHz ({prom:.0f}x local floor, "
                            f"post-edge)")
                    else:
                        notes.append(
                            f"{ch.display_label()}: no narrowband "
                            f"ringing stands out ({prom:.1f}x floor)")
        self.det_plot.setLabel("bottom", "time (display units)")
        self.lbl_det.setText(
            f"{int(m.sum()):,} full-resolution samples over "
            f"{x0:.5g}..{x1:.5g} (dt = {dt*1e9:.3g} ns). "
            + ("Ringing candidates - " + "; ".join(notes) if notes
               else "No >1 kHz component stands out."))

    # ------------------------------------------------------------------
    def _build_gpu_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        try:
            from gl_waterfall import GLWaterfall
            self._glw = GLWaterfall()
        except Exception as e:
            self._glw = None
            lbl = QLabel(
                "GPU renderer unavailable - install the OpenGL bindings "
                "inside your venv and restart:\n\n"
                "    pip install PyOpenGL PyOpenGL-accelerate\n\n"
                f"(detail: {e})")
            lbl.setWordWrap(True)
            v.addWidget(lbl)
            v.addStretch(1)
            return w
        row = QHBoxLayout()
        btn_cas = QPushButton("Cascade enabled channels (GPU)")
        btn_cas.setToolTip(
            "All checked channels from the main window, full resolution "
            "(min-max decimated to 200k vertices/trace), rendered on the "
            "GPU - smooth rotation even at millions of points.")
        btn_cas.clicked.connect(self._gpu_cascade)
        btn_surf = QPushButton("Demo surface (GPU)")
        btn_surf.clicked.connect(self._gpu_demo_surface)
        row.addWidget(btn_cas)
        row.addWidget(btn_surf)
        row.addStretch(1)
        v.addLayout(row)
        self.lbl_gpu = QLabel("Drag to orbit - scroll to zoom. Axes are "
                              "normalized to a unit box; ranges shown "
                              "here after plotting.")
        self.lbl_gpu.setWordWrap(True)
        v.addWidget(self.lbl_gpu)
        v.addWidget(self._glw.widget, 1)
        return w

    def _gpu_cascade(self):
        df = self._main_df()
        if df is None or self._glw is None:
            QMessageBox.information(self, "GPU 3D",
                                    "Load a CSV in the main window first.")
            return
        p = self.parent()
        chans = self._checked_channels() or \
            [ch for ch in getattr(p, "channels", []) if ch.enabled]
        if not chans:
            QMessageBox.information(self, "GPU 3D",
                                    "Tick channels in the Shot data 3D "
                                    "tab first.")
            return
        t = p._x()
        traces, colors = [], []
        for ch in chans:
            try:
                traces.append((ch.display_label(), p._channel_data(ch)))
                colors.append(ch.color)
            except Exception:
                continue
        t0, t1, ylo, yhi = self._glw.add_cascade(t, traces, colors)
        self.lbl_gpu.setText(
            f"{len(traces)} calibrated trace(s), {len(t):,} samples each "
            f"(decimated to <=200k vertices). x: {t0:.4g}..{t1:.4g} | "
            f"z: {ylo:.4g}..{yhi:.4g}.")

    def _gpu_demo_surface(self):
        if self._glw is None:
            return
        x, y, z = mexican_hat_surface(n=401)
        self._glw.add_surface(x[0], y[:, 0], z,
                              self.cmb_cmap.currentText())
        self.lbl_gpu.setText("Mexican-hat potential, 401x401 grid, "
                             "GPU-shaded.")

    # ------------------------------------------------------------------
    def _build_shot_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        btn_load = QPushButton("Load shot CSV...")
        btn_load.setToolTip(
            "Load an oscilloscope time-series CSV through the main app's "
            "read-only loader, then refresh this waterfall table.")
        btn_load.clicked.connect(self._load_shot_csv)
        btn_cols = QPushButton("Refresh columns")
        btn_cols.clicked.connect(self._refresh_columns)
        self.cmb_cmap2 = QComboBox()
        self.cmb_cmap2.addItems(COLORMAPS)
        btn_plot = QPushButton("Plot 3D waterfall")
        btn_plot.clicked.connect(self._plot_shot)
        row.addWidget(btn_load)
        row.addWidget(btn_cols)
        row.addWidget(QLabel("Colormap:"))
        row.addWidget(self.cmb_cmap2)
        row.addWidget(btn_plot)
        row.addStretch(1)
        v.addLayout(row)
        # calibration table - SAME channels as the main window: editing
        # gain/offset here writes back and refreshes both views
        self.tbl_cols = QTableWidget(0, 4)
        self.tbl_cols.setHorizontalHeaderLabels(
            ["Plot", "Column", "Gain", "Offset"])
        self.tbl_cols.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch)
        self.tbl_cols.setMaximumHeight(150)
        self.tbl_cols.itemChanged.connect(self._shot_table_edited)
        self._tbl_loading = False
        v.addWidget(self.tbl_cols)
        info = QLabel(
            "Channels and calibration are shared with the MAIN window: "
            "the formula/preset comes from the channel table there, and "
            "Gain/Offset edited here write back (both views refresh). "
            "Waterfall: x = time, depth = channel, height = calibrated "
            "value; ~3000 points per trace.")
        info.setWordWrap(True)
        v.addWidget(info)
        self.fig2 = Figure(figsize=(7, 6), tight_layout=True)
        self.canvas2 = FigureCanvasQTAgg(self.fig2)
        v.addWidget(self.canvas2, 1)
        self._refresh_columns()
        return w

    def _load_shot_csv(self, path: str | None = None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open oscilloscope shot CSV", "",
                "Data files (*.csv *.CSV *.txt *.tsv);;All files (*)")
        if not path:
            return False
        parent = self.parent()
        if not hasattr(parent, "open_csv_path"):
            QMessageBox.information(
                self, "Shot data 3D",
                "Standalone 3D mode cannot share shot calibration. "
                "Open this window from Scope Studio to load shot CSVs.")
            return False
        parent.open_csv_path(path)
        self._refresh_columns()
        self._set_tab_index(1)
        return True

    def _main_df(self):
        d = getattr(self.parent(), "data", None)
        return getattr(d, "df", None)

    def _refresh_columns(self):
        self._tbl_loading = True
        chans = getattr(self.parent(), "channels", []) or []
        self.tbl_cols.setRowCount(len(chans))
        for r, ch in enumerate(chans):
            chk = QTableWidgetItem("")
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(
                Qt.Checked if ch.enabled and
                "peak" not in ch.name.lower() else Qt.Unchecked)
            name = QTableWidgetItem(ch.name)
            name.setFlags(Qt.ItemIsEnabled)
            name.setToolTip(f"formula: {ch.formula}   "
                            f"preset: {ch.preset_name}")
            gain = QTableWidgetItem(f"{ch.gain:g}")
            off = QTableWidgetItem(f"{ch.offset:g}")
            self.tbl_cols.setItem(r, 0, chk)
            self.tbl_cols.setItem(r, 1, name)
            self.tbl_cols.setItem(r, 2, gain)
            self.tbl_cols.setItem(r, 3, off)
        self._tbl_loading = False
        # keep the V-I channel pickers in sync (tab may not exist yet
        # during construction)
        if hasattr(self, "cmb_vi_x"):
            labels = [ch.display_label() for ch in chans]
            for cmb in (self.cmb_vi_x, self.cmb_vi_y):
                cur = cmb.currentText()
                cmb.clear()
                cmb.addItems(labels)
                if cur in labels:
                    cmb.setCurrentIndex(labels.index(cur))
            # sensible defaults: X = first current (A), Y = first volts
            ia = next((i for i, l in enumerate(labels) if "(A)" in l), 0)
            iv = next((i for i, l in enumerate(labels) if "(V)" in l),
                      min(1, max(len(labels) - 1, 0)))
            if self.cmb_vi_x.currentText() not in labels or \
                    self.cmb_vi_x.currentIndex() < 0:
                self.cmb_vi_x.setCurrentIndex(ia)
            if self.cmb_vi_y.currentText() not in labels or \
                    self.cmb_vi_y.currentIndex() < 0:
                self.cmb_vi_y.setCurrentIndex(iv)

    def _shot_table_edited(self, item):
        """Write gain/offset edits back to the shared Channel objects so
        the main 2D plot and the 3D views stay consistent."""
        if self._tbl_loading or item.column() not in (2, 3):
            return
        p = self.parent()
        chans = getattr(p, "channels", []) or []
        r = item.row()
        if r >= len(chans):
            return
        try:
            val = float(item.text())
        except ValueError:
            self._refresh_columns()       # revert bad input
            return
        if item.column() == 2:
            chans[r].gain = val
        else:
            chans[r].offset = val
        try:                              # sync the main window's table
            col = 4 if item.column() == 2 else 5
            p.table.item(r, col).setText(f"{val:g}")
        except Exception:
            pass
        try:
            p._transform_cache.clear()
            p.refresh_plot()
        except Exception:
            pass

    def _checked_channels(self):
        p = self.parent()
        chans = getattr(p, "channels", []) or []
        out = []
        for r in range(self.tbl_cols.rowCount()):
            it = self.tbl_cols.item(r, 0)
            if it is not None and it.checkState() == Qt.Checked \
                    and r < len(chans):
                out.append(chans[r])
        return out

    def _plot_shot(self):
        p = self.parent()
        if self._main_df() is None:
            QMessageBox.information(self, "Shot data 3D",
                                    "Load a CSV in the main window first, "
                                    "then click Refresh columns.")
            return
        chans = self._checked_channels()
        if not chans:
            QMessageBox.information(self, "Shot data 3D",
                                    "Tick at least one channel.")
            return
        t = p._x()                       # display units, same as 2D plot
        cmap = cm.get_cmap(self.cmb_cmap2.currentText())
        self.fig2.clf()
        ax = self.fig2.add_subplot(111, projection="3d")
        labels = []
        for k, ch in enumerate(chans):
            try:                         # full calibration incl. filter
                y = p._channel_data(ch)
            except Exception:
                continue
            step = max(1, len(y) // 3000)
            ts, ys = t[::step], y[::step]
            frac = k / max(1, len(chans) - 1) if len(chans) > 1 else 0.35
            ax.plot(ts, np.full(ts.shape, k), ys,
                    color=cmap(frac), lw=1.0)
            labels.append(ch.display_label())
        ax.set_xlabel("Time (ms)")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_zlabel("calibrated value")
        self.canvas2.draw_idle()

    # ------------------------------------------------------------------
    def _load_demo(self, idx: int):
        self._axis_labels = ("x", "y", "z")
        if idx == 0:
            self._data = mexican_hat_surface()
            self._color = None
        elif idx == 1:
            self._data = gaussian_surface()
            self._color = None
        else:
            from synthetic_vi_didt import (
                synthetic_current_surface,
                synthetic_didt_surface,
                synthetic_voltage_surface,
            )
            if idx == 2:
                self._data, self._color = synthetic_current_surface()
                self._axis_labels = (
                    "time (ms)", "drive voltage (V)", "current (A)")
            elif idx == 3:
                self._data, self._color = synthetic_didt_surface()
                self._axis_labels = (
                    "time (ms)", "drive voltage (V)", "dI/dt (A/s)")
            else:
                self._data, self._color = synthetic_voltage_surface()
                self._axis_labels = (
                    "time (ms)", "drive voltage (V)", "L*dI/dt (V)")
        self.redraw()

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open surface CSV", "",
            "CSV files (*.csv *.CSV);;All (*)")
        if not path:
            return
        try:
            df = load_table(path)
        except Exception as e:
            QMessageBox.warning(self, "3D view",
                                f"Could not read this file:\n{e}")
            return
        cols = {str(c).strip().lower(): c for c in df.columns}
        if not {"x", "y", "z"} <= set(cols) and looks_like_time_series(df):
            route = self._time_series_route_dialog(
                "This looks like an oscilloscope time-series CSV, not a "
                "surface/matrix file.")
            if route == "shot":
                self._load_shot_csv(path)
                return
            if route == "cancel":
                return
        try:
            if {"x", "y", "z"} <= set(cols):
                x, y, z, c = pivot_grid(df, cols["x"], cols["y"],
                                        cols["z"], cols.get("c"))
            else:
                # let the user map columns to axes (works for any
                # column names / any file the robust loader can read)
                picks = self._ask_columns(list(df.columns))
                if picks is None:
                    return
                xc, yc, zc, cc = picks
                x, y, z, c = pivot_grid(df, xc, yc, zc, cc)
        except Exception as e:
            route = self._time_series_route_dialog(
                f"Could not grid this file:\n{e}")
            if route == "shot":
                self._load_shot_csv(path)
            return
        self._data, self._color = (x, y, z), c
        self._axis_labels = ("x", "y", "z")
        self.redraw()

    def _time_series_route_dialog(self, detail: str) -> str:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("3D CSV routing")
        msg.setText(detail)
        msg.setInformativeText(
            "Surface files use x,y,z grids. Scope shots are better shown "
            "as 3D waterfalls in the Shot data 3D tab.")
        btn_shot = msg.addButton("Open in Shot data 3D",
                                 QMessageBox.ButtonRole.AcceptRole)
        btn_surface = msg.addButton("Try surface mapping anyway",
                                    QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_shot:
            return "shot"
        if clicked == btn_surface:
            return "surface"
        return "cancel"

    def _ask_columns(self, columns: list[str]):
        """Dialog: map file columns to x / y / z (+ optional color)."""
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox,
                                       QFormLayout)
        dlg = QDialog(self)
        dlg.setWindowTitle("Choose surface axes")
        form = QFormLayout(dlg)
        boxes = {}
        for i, axis in enumerate(("x", "y", "z")):
            cb = QComboBox()
            cb.addItems([str(c) for c in columns])
            cb.setCurrentIndex(min(i, len(columns) - 1))
            form.addRow(f"{axis} column:", cb)
            boxes[axis] = cb
        cbc = QComboBox()
        cbc.addItems(["(none)"] + [str(c) for c in columns])
        form.addRow("color (4th dim):", cbc)
        bb = QDialogButtonBox(QDialogButtonBox.Ok |
                              QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return None
        cc = cbc.currentText()
        return (boxes["x"].currentText(), boxes["y"].currentText(),
                boxes["z"].currentText(), None if cc == "(none)" else cc)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export 3D figure", "surface.png",
            "PNG (*.png);;SVG vector (*.svg)")
        if path:
            self.fig.savefig(path, dpi=600)

    # ------------------------------------------------------------------
    def redraw(self):
        x, y, z = self._data
        cmap = cm.get_cmap(self.cmb_cmap.currentText())
        self.fig.clf()
        ax = self.fig.add_subplot(111, projection="3d")
        if self._color is not None and self._color.shape == z.shape:
            # 4th dimension: color carries C(x,y), height carries z
            norm = (self._color - np.nanmin(self._color)) / max(
                np.nanmax(self._color) - np.nanmin(self._color), 1e-12)
            surf = ax.plot_surface(x, y, z, facecolors=cmap(norm),
                                   rstride=1, cstride=1,
                                   linewidth=0, antialiased=True)
            m = cm.ScalarMappable(cmap=cmap)
            m.set_array(self._color)
            self.fig.colorbar(m, ax=ax, shrink=0.6, label="C (4th dim)")
        else:
            surf = ax.plot_surface(x, y, z, cmap=cmap, rstride=1,
                                   cstride=1, linewidth=0,
                                   antialiased=True)
            self.fig.colorbar(surf, ax=ax, shrink=0.6, label="z")
        ax.set_xlabel(self._axis_labels[0])
        ax.set_ylabel(self._axis_labels[1])
        ax.set_zlabel(self._axis_labels[2])
        self.canvas.draw_idle()


def main() -> int:
    app = QApplication(sys.argv)
    win = Surface3DWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
