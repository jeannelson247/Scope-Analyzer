#!/usr/bin/env python3
"""
Scope Analyzer — fast oscilloscope CSV viewer & publication-figure exporter
=========================================================================
Built for tokamak current-driver testing (TF-coil module commissioning).

  * Open large scope CSVs (preamble/delimiter auto-detected, float32 memory)
  * Tick the channels to plot; assign each to the Left or Right y-axis
  * Per-channel conversion presets (Pearson / bus-bar / Hall monitors): V → A
  * Smooth zoom & pan even on millions of points (peak-preserving decimation)
  * Left/right zero-alignment, optional top (4th) axis with its own scale
  * Manual or auto X / Y-left / Y-right ranges
  * Stats over the visible window: peak, min, mean, RMS, pk-pk, t@peak
  * Export Nature-style figures: SVG (vector, editable text) or 600-dpi JPG
  * Optional local-AI shot summary via Ollama (see ai_assistant.py)

Run:  python app.py
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal, QMimeData, QTimer, QUrl
from PySide6.QtGui import QColor, QAction, QTextCursor, QDesktopServices

# Start in the 2D scope view. The 3D/V-I/FFT views now live as in-window
# tabs instead of opening automatically in a separate window.
SHOW_3D_AT_LAUNCH = False

# Jean's rig defaults (4 busbar monitors x ~1500 A soft-saturation, summed
# at gain 4 -> 6000 A; Pearson trustworthy before core walk-off ~5 ms).
# The one-click shot pipeline uses these; edit here when the rig changes.
USER_DEFAULTS = {
    "sat_level": 6000.0,
    "ref_end_ms": 5.0,
}
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QTableWidget, QTableWidgetItem, QComboBox,
    QCheckBox, QDoubleSpinBox, QLineEdit, QSpinBox, QSplitter, QHeaderView,
    QColorDialog, QMessageBox, QPlainTextEdit, QScrollArea, QFormLayout,
    QGridLayout, QSizePolicy, QDialog, QDialogButtonBox, QTabWidget,
)

from csv_loader import load_csv, LoadedData, minmax_decimate
from data_quality import quality_report
from data_session import DataSession
from shot_metadata import ensure_sidecar
from nature_export import ExportTrace, ExportOptions, export_figure
from signal_tools import FormulaError, evaluate_formula, lowpass
from plot_render import make_curve, enable_view_downsampling
from calibration import fit_forced_origin_gain, format_gain_fit, CalibrationError

HERE = os.path.dirname(os.path.abspath(__file__))

PALETTES = {
    "Wong / Nature": [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7",
        "#56B4E9", "#D55E00", "#F0E442", "#000000",
    ],
    "Nature Physics (NPG)": [
        "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
        "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
    ],
    "Tokamak dual-axis": [
        "#000000", "#0072B2", "#D55E00", "#009E73",
        "#CC79A7", "#56B4E9", "#E69F00", "#7F7F7F",
    ],
}

FORMULA_HINT = (
    "Formula helpers: `x`, `t` (s), `t_ms`, `baseline()`, `lowpass()`, "
    "`integrate()`, `movmean()`, `samples()`, `gradient()`. Gain/offset "
    "apply after the formula. Leave Label blank to auto-use column name + "
    "unit."
)

CHAT_SYSTEM_PROMPT = (
    "You are a local analysis assistant for tokamak current-driver testing. "
    "Use the visible waveform summaries and any retrieved paper excerpts to "
    "answer the user's question concisely and numerically. Mention caveats "
    "when the visible data is insufficient. Do not invent channels, papers, "
    "or values. Hard rule: original CSV files are immutable read-only "
    "measurements. You may request reversible display changes, in-memory "
    "overlays, or draft tools, but never ask to overwrite or modify a "
    "source CSV."
)

# Action protocol: lets the model reformat the plot or run deterministic
# tools (stats / anomaly scan) by appending a JSON block to its reply.
from chat_actions import ACTION_SCHEMA  # noqa: E402
CHAT_SYSTEM_PROMPT += ACTION_SCHEMA

pg.setConfigOptions(antialias=False, useOpenGL=False,
                    background="w", foreground="k")


def load_presets() -> dict:
    fallback = {
        "Raw (no conversion)": {
            "gain": 1.0, "offset": 0.0, "unit": "V", "formula": "x"
        }
    }
    try:
        with open(os.path.join(HERE, "presets.json")) as f:
            raw = json.load(f)
    except Exception:
        return fallback
    presets = {}
    for name, spec in raw.items():
        presets[name] = {
            "gain": float(spec.get("gain", 1.0)),
            "offset": float(spec.get("offset", 0.0)),
            "unit": spec.get("unit", ""),
            "formula": (spec.get("formula") or "x").strip() or "x",
        }
    if "Raw (no conversion)" not in presets:
        presets["Raw (no conversion)"] = fallback["Raw (no conversion)"]
    return presets


@dataclass
class Channel:
    name: str
    enabled: bool = False
    axis: str = "left"            # "left" | "right"
    gain: float = 1.0
    offset: float = 0.0
    label: str = ""
    color: str = "#0072B2"
    formula: str = "x"
    unit: str = ""
    preset_name: str = "Raw (no conversion)"

    def display_label(self) -> str:
        if self.label.strip():
            return self.label.strip()
        if self.unit:
            return f"{self.name} ({self.unit})"
        return self.name

    def convert(self, y: np.ndarray, t_s: np.ndarray,
                t_ms: np.ndarray) -> np.ndarray:
        out = evaluate_formula(self.formula, y, t_s, t_ms)
        if self.gain != 1.0 or self.offset != 0.0:
            out = out * self.gain + self.offset
        return out


class TransformedAxis(pg.AxisItem):
    """Top axis that shows bottom-axis values transformed by scale·x+offset."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.scale_f, self.offset_f = 1.0, 0.0

    def tickStrings(self, values, scale, spacing):
        return [f"{v * self.scale_f + self.offset_f:g}" for v in values]


class ModelWorker(QThread):
    done = Signal(str)

    def __init__(self, prompt: str, model: str, backend: str,
                 system_prompt: str, max_tokens: int = 768):
        super().__init__()
        self.prompt = prompt
        self.model = model
        self.backend = backend
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def run(self):
        from ai_assistant import ask_model
        self.done.emit(ask_model(self.prompt, self.model, self.backend,
                                 system_prompt=self.system_prompt,
                                 max_tokens=self.max_tokens))


class RouterWorker(QThread):
    """Tier-1 of the two-tier chat: a tiny model (DEFAULT_ROUTER_MODEL)
    decides in ~0.2 s whether the user's message is a tool request.
    Emits the router's JSON string; '{"run": "none"}' or any parse
    failure means 'not a tool request' and the heavy model takes over."""
    done = Signal(str)

    def __init__(self, question: str, model: str,
                 backend: str = "ollama"):
        super().__init__()
        self.question = question
        self.model = model
        self.backend = backend

    def run(self):
        from ai_assistant import route_action
        self.done.emit(route_action(self.question, model=self.model,
                                    backend=self.backend))


class PaperIndexWorker(QThread):
    done = Signal(object, str)

    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder

    def run(self):
        from paper_index import build_index
        try:
            index = build_index(self.folder)
            msg = (f"Indexed {len(index.files)} files into "
                   f"{len(index.chunks)} searchable chunks.")
            if index.skipped:
                msg += f" Skipped {len(index.skipped)} file(s)."
            self.done.emit(index, msg)
        except Exception as exc:
            self.done.emit(None, f"Paper indexing failed: {exc}")


# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from version import __version__ as _scope_version
        self.setWindowTitle(
            f"Scope Analyzer {_scope_version} — CSV viewer & Nature-figure export")
        # never open wider than the screen (a fixed 1760 px clipped the AI
        # panel off the right edge on 14" MacBooks)
        scr = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1760, scr.width() - 40),
                    min(920, scr.height() - 60))

        self.data: LoadedData | None = None
        self.session: DataSession | None = None
        self.data_quality = None
        self.shot_metadata = None
        self.shot_metadata_path = None
        self.channels: list[Channel] = []
        self.presets = load_presets()
        self.curves: dict[str, pg.PlotDataItem] = {}
        self._transform_cache: dict[tuple, np.ndarray] = {}
        self._building_table = False
        self._model_worker = None
        self._paper_worker = None
        self._paper_index = None
        self._pending_sources: list[str] = []
        self._chat_history: list[dict[str, str]] = []
        self._ai_trace_events: list[str] = []
        self._undo_stack: list[tuple[str, dict]] = []

        self._build_plot()
        self._build_controls()
        self._build_ai_panel()
        self._build_workspace_tabs()

        splitter = QSplitter()
        splitter.addWidget(self.controls_scroll)
        splitter.addWidget(self.workspace_tabs)
        splitter.addWidget(self.ai_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([500, 880, 420])
        self.setCentralWidget(splitter)

        open_act = QAction("Open CSV…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self.open_csv)
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_act)

        example_act = QAction("Load example shot", self)
        example_act.setToolTip("Load a bundled synthetic shot to explore the "
                               "app without your own data.")
        example_act.triggered.connect(self.load_example_shot)
        file_menu.addAction(example_act)

        save_an_act = QAction("Save analyzed copy…", self)
        save_an_act.setShortcut("Ctrl+Shift+S")
        save_an_act.triggered.connect(self.save_analyzed_copy)
        file_menu.addAction(save_an_act)

        obs_act = QAction("Export session note to Obsidian…", self)
        obs_act.setShortcut("Ctrl+Shift+N")
        obs_act.triggered.connect(self.export_obsidian_note)
        file_menu.addAction(obs_act)

        add_ovl_act = QAction("Add overlay shots…", self)
        add_ovl_act.setShortcut("Ctrl+Shift+O")
        add_ovl_act.triggered.connect(self.add_overlay_shots)
        clr_ovl_act = QAction("Clear overlay shots", self)
        clr_ovl_act.triggered.connect(self.clear_overlay_shots)
        file_menu.addAction(add_ovl_act)
        file_menu.addAction(clr_ovl_act)

        view3d_act = QAction("3D surface view…", self)
        view3d_act.setShortcut("Ctrl+3")
        view3d_act.triggered.connect(self._open_3d_view)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(view3d_act)

        copy_img_act = QAction("Copy figure as image", self)
        copy_img_act.setShortcut("Ctrl+Shift+C")
        copy_img_act.triggered.connect(lambda: self.copy_figure("image"))
        copy_svg_act = QAction("Copy figure as SVG (vector)", self)
        copy_svg_act.triggered.connect(lambda: self.copy_figure("svg"))
        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(copy_img_act)
        edit_menu.addAction(copy_svg_act)

        # Keep the startup view calm; 3D/V-I/FFT are available as tabs.
        if SHOW_3D_AT_LAUNCH:
            QTimer.singleShot(600, self._open_3d_view)

    def _plot_mouse_clicked(self, ev):
        if not ev.double():
            return
        if ev.modifiers() & Qt.ShiftModifier:      # power-user shortcut
            self.copy_figure("svg")
            return
        # dropdown at the cursor: pick the clipboard format explicitly
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addAction("Copy as PNG",
                       lambda: self.copy_figure("png"))
        menu.addAction("Copy as JPG",
                       lambda: self.copy_figure("jpg"))
        menu.addAction("Copy as SVG (vector)",
                       lambda: self.copy_figure("svg"))
        menu.addSeparator()
        menu.addAction("Save analyzed copy…", self.save_analyzed_copy)
        menu.exec(QCursor.pos())

    def copy_figure(self, fmt: str = "image"):
        """Copy the current plot to the clipboard.
        'image' puts a high-resolution bitmap on the clipboard (paste as
        PNG/JPG/TIFF - the receiving app picks the encoding; that is how
        clipboards work, the bitmap itself is format-neutral).
        'svg' puts vector XML on the clipboard (image/svg+xml + plain
        text), pasteable into Illustrator/Inkscape/editors."""
        import pyqtgraph.exporters as pgex
        try:
            if fmt == "svg":
                import os
                import tempfile
                ex = pgex.SVGExporter(self.pi)
                tmp = tempfile.NamedTemporaryFile(suffix=".svg",
                                                  delete=False)
                tmp.close()
                ex.export(tmp.name)
                with open(tmp.name, "rb") as fh:
                    data = fh.read()
                os.unlink(tmp.name)
                md = QMimeData()
                md.setData("image/svg+xml", data)
                md.setText(data.decode("utf-8", "replace"))
                QApplication.clipboard().setMimeData(md)
                self.statusBar().showMessage(
                    "Figure copied as SVG vector.", 5000)
            else:
                from PySide6.QtCore import QBuffer, QIODevice
                ex = pgex.ImageExporter(self.pi)
                # ~2x retina resolution for crisp pastes into docs/slides
                ex.parameters()["width"] = 2400
                img = ex.export(toBytes=True)
                md = QMimeData()
                if fmt in ("png", "jpg"):
                    # encode to the REQUESTED format and put those bytes
                    # on the clipboard under the right mime type, with a
                    # plain bitmap fallback for picky apps
                    buf = QBuffer()
                    buf.open(QIODevice.WriteOnly)
                    img.save(buf, "PNG" if fmt == "png" else "JPG", 95)
                    md.setData("image/png" if fmt == "png"
                               else "image/jpeg", buf.data())
                md.setImageData(img)
                QApplication.clipboard().setMimeData(md)
                self.statusBar().showMessage(
                    f"Figure copied as {fmt.upper() if fmt != 'image' else 'bitmap'} "
                    f"(paste anywhere).", 5000)
        except Exception as e:
            self.statusBar().showMessage(f"Copy failed: {e}", 8000)

    def save_analyzed_copy(self):
        """Write a NEW file: every original column untouched, plus one
        '<channel>_analyzed' column per enabled channel containing the
        fully calibrated/processed trace (formula, gain, offset, filter)
        and the RLC reconstruction if present. The source CSV is never
        modified (hash-verified policy)."""
        if self.data is None:
            QMessageBox.information(self, "Save analyzed copy",
                                    "Load a file first.")
            return
        import os
        src = self.data.path
        suggest = os.path.splitext(src)[0] + "_analyzed.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save analyzed copy", suggest, "CSV (*.csv)")
        if not path:
            return
        if os.path.abspath(path) == os.path.abspath(src):
            QMessageBox.warning(self, "Save analyzed copy",
                                "Refusing to overwrite the original "
                                "measurement file - choose a new name.")
            return
        try:
            df = self.data.df.copy()
            for ch in self.channels:
                if not ch.enabled:
                    continue
                try:
                    df[f"{ch.name}_analyzed"] = self._channel_data(ch)
                except Exception:
                    continue
            cv = getattr(self, "_recon_overlay", None)
            if cv:                       # reconstruction on its own grid
                xq = self._x()
                df["RLC_reconstruction"] = np.interp(
                    xq, np.asarray(cv["t"]), np.asarray(cv["y"]))
            df.to_csv(path, index=False)
            self.statusBar().showMessage(
                f"Analyzed copy saved: {os.path.basename(path)} "
                f"(original untouched).", 8000)
        except Exception as e:
            QMessageBox.warning(self, "Save analyzed copy", str(e))

    def _build_workspace_tabs(self):
        """Build the browser-style central workspace.

        The existing 3D implementation remains the owner of its widgets and
        callbacks; we move those widgets into the main tab bar and keep the
        owner object alive in ``self._win3d``.
        """
        self.workspace_tabs = QTabWidget()
        self.workspace_tabs.setDocumentMode(True)
        self.workspace_tabs.setMovable(False)
        self.workspace_tabs.addTab(self.plot_area, "2D plot")
        self._workspace_tab_by_key = {"2d": 0}
        self._surface_tab_offset = 1

        try:
            from surface3d import Surface3DWindow
        except Exception as e:
            self._win3d = None
            self.workspace_tabs.addTab(
                QLabel(f"3D module failed to load:\n{e}"), "3D unavailable")
            return

        self._win3d = Surface3DWindow(self)
        source_tabs = self._win3d.tabs
        labels = []
        while source_tabs.count():
            widget = source_tabs.widget(0)
            label = source_tabs.tabText(0)
            source_tabs.removeTab(0)
            idx = self.workspace_tabs.addTab(widget, label)
            labels.append(label)
            self._workspace_tab_by_key[label.lower()] = idx

        # The 3D owner methods sometimes switch tabs after loading data. Point
        # those switches at the embedded workspace and account for the 2D tab.
        self._win3d.setCentralWidget(QWidget())
        self._win3d.tabs = self.workspace_tabs
        self._win3d._embedded_tab_offset = self._surface_tab_offset

    def _refresh_embedded_3d(self):
        win = getattr(self, "_win3d", None)
        if win is not None and hasattr(win, "_refresh_columns"):
            try:
                win._refresh_columns()
            except Exception:
                pass

    def _open_3d_view(self, tab_index: int = 0):
        """Switch to an embedded 3D tab."""
        tabs = getattr(self, "workspace_tabs", None)
        if tabs is None:
            return
        target = self._surface_tab_offset + int(tab_index)
        if 0 <= target < tabs.count():
            tabs.setCurrentIndex(target)
            self.raise_()
            self.activateWindow()
            self._refresh_embedded_3d()
        else:
            QMessageBox.warning(self, "3D view", "Requested 3D tab is not available.")

    def _mode_selected(self, idx: int):
        """Control-row Mode launcher for the central workspace tabs."""
        if idx == 0:
            self.workspace_tabs.setCurrentIndex(0)
            return
        if idx == 5:
            self.run_anomaly_scan()
        else:
            # dropdown index -> Surface3DWindow tab index
            tab = {1: 0, 2: 1, 3: 3, 4: 4}.get(idx, 0)
            self._open_3d_view(tab)
        self.cmb_mode.blockSignals(True)
        self.cmb_mode.setCurrentIndex(0)
        self.cmb_mode.blockSignals(False)

    # ------------------------------------------------------------- plot ----
    def _build_plot(self):
        self.plot_widget = pg.PlotWidget()
        self.pi = self.plot_widget.getPlotItem()
        self.pi.showGrid(x=True, y=True, alpha=0.15)
        self.pi.setLabel("bottom", "Time (ms)")
        self.pi.setLabel("left", "Current (A)")
        # MATLAB-style: double-click the plot -> figure straight to the
        # clipboard (high-res bitmap; Shift+double-click for SVG vector)
        self.plot_widget.scene().sigMouseClicked.connect(
            self._plot_mouse_clicked)

        # ----- mouse-mode toolbar (handlebar) -----
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 0)

        # analysis-mode launcher: one discoverable place to reach every view.
        bar.addWidget(QLabel("Mode:"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["2D plot", "3D surface", "Shot 3D overlay",
                                "V-I map", "Detail + FFT", "Anomaly scan"])
        self.cmb_mode.setToolTip(
            "Jump to an analysis tab. Anomaly scan runs a deterministic "
            "scan in the side chat.")
        self.cmb_mode.activated.connect(self._mode_selected)
        bar.addWidget(self.cmb_mode)
        bar.addSpacing(16)

        self.mode_buttons: dict[str, QPushButton] = {}
        for key, text, tip in [
            ("pan",  "✋ Pan",      "Drag to pan, wheel zooms both axes"),
            ("box",  "⬚ Box zoom", "Drag a rectangle to zoom into it"),
            ("xy",   "⤡ Zoom XY",  "Wheel/drag zooms both axes"),
            ("x",    "↔ Zoom X",   "Wheel/drag affects the X axis only"),
            ("y",    "↕ Zoom Y",   "Wheel/drag affects the Y axis only"),
        ]:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _, k=key: self._set_mouse_mode(k))
            bar.addWidget(b)
            self.mode_buttons[key] = b
        bar.addSpacing(12)
        for text, fn, tip in [
            ("＋", lambda: self._zoom_step(1 / 1.25), "Zoom in"),
            ("－", lambda: self._zoom_step(1.25), "Zoom out"),
            ("⟲ Reset", self._reset_zoom, "Autoscale everything (key: A)"),
        ]:
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch(1)

        self.plot_area = QWidget()
        va = QVBoxLayout(self.plot_area)
        va.setContentsMargins(0, 0, 0, 0)
        va.setSpacing(2)
        va.addLayout(bar)
        va.addWidget(self.plot_widget)

        # Right-axis viewbox (linked X)
        self.vb_right = pg.ViewBox()
        self.pi.scene().addItem(self.vb_right)
        self.right_axis = self.pi.getAxis("right")
        self.right_axis.linkToView(self.vb_right)
        self.vb_right.setXLink(self.pi.vb)
        self.pi.vb.sigResized.connect(self._sync_right_geometry)
        self.vb_right.setMouseEnabled(x=False, y=True)

        # Top axis (transformed)
        self.top_axis = TransformedAxis(orientation="top")
        self.pi.layout.removeItem(self.pi.getAxis("top"))
        self.pi.axes["top"]["item"] = self.top_axis
        self.pi.layout.addItem(self.top_axis, 1, 1)
        self.top_axis.linkToView(self.pi.vb)
        self.top_axis.hide()

        self.right_axis.hide()
        self.pi.vb.sigYRangeChanged.connect(self._maybe_align_zero)
        self._set_mouse_mode("pan")

        # "A" autoscales — scoped to the plot (focus follows a click there)
        # so it never fires while typing in formula / label / range fields.
        from PySide6.QtGui import QShortcut, QKeySequence
        self.plot_widget.setFocusPolicy(Qt.StrongFocus)
        sc_reset = QShortcut(QKeySequence(Qt.Key_A), self.plot_widget)
        sc_reset.setContext(Qt.WidgetWithChildrenShortcut)
        sc_reset.activated.connect(self._reset_zoom)

    def _zoom_step(self, factor: float):
        mode = getattr(self, "_mouse_mode", "pan")
        sx = factor if mode in ("pan", "xy", "x", "box") else 1.0
        sy = factor if mode in ("pan", "xy", "y", "box") else 1.0
        self.pi.vb.scaleBy((sx, sy))

    def _sync_right_geometry(self):
        self.vb_right.setGeometry(self.pi.vb.sceneBoundingRect())
        self.vb_right.linkedViewChanged(self.pi.vb, self.vb_right.XAxis)

    # ----------------------------------------------------- mouse modes -----
    def _set_mouse_mode(self, mode: str):
        for key, btn in getattr(self, "mode_buttons", {}).items():
            btn.blockSignals(True)
            btn.setChecked(key == mode)
            btn.blockSignals(False)
        vb = self.pi.vb
        if mode == "box":
            vb.setMouseMode(pg.ViewBox.RectMode)
            vb.setMouseEnabled(x=True, y=True)
        else:
            vb.setMouseMode(pg.ViewBox.PanMode)
            vb.setMouseEnabled(x=mode in ("pan", "x", "xy"),
                               y=mode in ("pan", "y", "xy"))
        # right-axis viewbox follows the same Y behaviour
        self.vb_right.setMouseEnabled(x=False,
                                      y=mode in ("pan", "y", "xy", "box"))
        self._mouse_mode = mode

    # --------------------------------------------------------- controls ----
    def _build_controls(self):
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setSpacing(8)

        # --- file ---
        gf = QGroupBox("Data file")
        hf = QHBoxLayout(gf)
        self.btn_open = QPushButton("Open CSV…")
        self.btn_open.setMinimumWidth(120)
        self.btn_open.setToolTip(
            "Open a scope CSV. Scope Analyzer reads the source file only; "
            "formulas, filters, overlays, and exports are separate views.")
        self.btn_open.clicked.connect(self.open_csv)
        self.lbl_file = QLabel("No file loaded — Open CSV (⌘O), or try "
                               "File ▸ Load example shot")
        self.lbl_file.setWordWrap(True)
        hf.addWidget(self.btn_open)
        hf.addWidget(self.lbl_file, 1)
        v.addWidget(gf)

        # --- plot style ---
        gp = QGroupBox("Plot style")
        fp = QFormLayout(gp)
        self.cmb_palette = QComboBox()
        self.cmb_palette.addItems(list(PALETTES))
        self.cmb_palette.setCurrentText("Wong / Nature")
        btn_palette = QPushButton("Apply palette to channels")
        btn_palette.clicked.connect(self._apply_palette_to_channels)
        fp.addRow("Palette:", self.cmb_palette)
        fp.addRow(btn_palette)
        v.addWidget(gp)

        # --- channels ---
        gc = QGroupBox("Channels  (tick → plot; preset converts V → A)")
        vc = QVBoxLayout(gc)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Plot", "Column", "Axis", "Preset", "Gain", "Offset",
             "Formula", "Label"])
        hdr = self.table.horizontalHeader()
        for i, w in enumerate([36, 110, 58, 170, 70, 70, 230, 120]):
            self.table.setColumnWidth(i, w)
        hdr.setSectionResizeMode(6, QHeaderView.Stretch)
        hdr.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.setMinimumHeight(240)
        self.table.cellChanged.connect(self._table_edited)
        self.table.verticalHeader().sectionDoubleClicked.connect(
            self._pick_color)
        vc.addWidget(self.table)
        vc.addWidget(QLabel(
            "Tip: double-click a row number to change trace colour."))
        lbl_formula = QLabel(FORMULA_HINT)
        lbl_formula.setWordWrap(True)
        vc.addWidget(lbl_formula)
        btn_formula = QPushButton("Formula editor / save preset...")
        btn_formula.clicked.connect(self._open_formula_dialog)
        vc.addWidget(btn_formula)
        v.addWidget(gc)

        # --- X column + axis ranges ---
        ga = QGroupBox("Axes")
        grid = QGridLayout(ga)
        r = 0
        grid.addWidget(QLabel("X column:"), r, 0)
        self.cmb_x = QComboBox()
        self.cmb_x.currentIndexChanged.connect(self._x_column_changed)
        grid.addWidget(self.cmb_x, r, 1)
        grid.addWidget(QLabel("X ×"), r, 2)
        self.spn_xscale = QDoubleSpinBox()
        self.spn_xscale.setDecimals(6)
        self.spn_xscale.setRange(1e-12, 1e12)
        self.spn_xscale.setValue(1000.0)   # s → ms by default
        self.spn_xscale.setToolTip("Multiply X (e.g. 1000 converts s → ms)")
        self.spn_xscale.valueChanged.connect(self.refresh_plot)
        grid.addWidget(self.spn_xscale, r, 3)
        r += 1

        def lim_row(label):
            nonlocal r
            grid.addWidget(QLabel(label), r, 0)
            lo, hi = QLineEdit(), QLineEdit()
            lo.setPlaceholderText("min")
            hi.setPlaceholderText("max")
            btn = QPushButton("Set")
            auto = QPushButton("Auto")
            grid.addWidget(lo, r, 1)
            grid.addWidget(hi, r, 2)
            grid.addWidget(btn, r, 3)
            grid.addWidget(auto, r, 4)
            r += 1
            return lo, hi, btn, auto

        self.x_lo, self.x_hi, bx, ax_ = lim_row("X range:")
        self.yl_lo, self.yl_hi, byl, ayl = lim_row("Y-left:")
        self.yr_lo, self.yr_hi, byr, ayr = lim_row("Y-right:")
        bx.clicked.connect(lambda: self._set_range("x"))
        byl.clicked.connect(lambda: self._set_range("yl"))
        byr.clicked.connect(lambda: self._set_range("yr"))
        ax_.clicked.connect(lambda: self.pi.vb.autoRange())
        ayl.clicked.connect(lambda: self.pi.vb.enableAutoRange(y=True))
        ayr.clicked.connect(lambda: self.vb_right.enableAutoRange(y=True))

        self.chk_zero = QCheckBox("Align zeros of left && right y-axes")
        self.chk_zero.setChecked(True)
        self.chk_zero.toggled.connect(self._maybe_align_zero)
        grid.addWidget(self.chk_zero, r, 0, 1, 5); r += 1

        self.chk_top = QCheckBox("Top axis")
        self.chk_top.toggled.connect(self.refresh_plot)
        grid.addWidget(self.chk_top, r, 0)
        grid.addWidget(QLabel("scale ×"), r, 1, Qt.AlignRight)
        self.spn_topscale = QDoubleSpinBox()
        self.spn_topscale.setDecimals(6)
        self.spn_topscale.setRange(-1e12, 1e12)
        self.spn_topscale.setValue(1.0)
        self.spn_topscale.valueChanged.connect(self.refresh_plot)
        grid.addWidget(self.spn_topscale, r, 2)
        self.ed_toplabel = QLineEdit()
        self.ed_toplabel.setPlaceholderText("top-axis label")
        self.ed_toplabel.editingFinished.connect(self.refresh_plot)
        grid.addWidget(self.ed_toplabel, r, 3, 1, 2); r += 1

        for cap, attr, default in [("X label:", "ed_xlabel", "Time (ms)"),
                                   ("Y-left:", "ed_yllabel", "Current (A)"),
                                   ("Y-right:", "ed_yrlabel",
                                    "Control Signal (V)"),
                                   ("Title:", "ed_title", "")]:
            grid.addWidget(QLabel(cap), r, 0)
            ed = QLineEdit(default)
            ed.editingFinished.connect(self.refresh_plot)
            setattr(self, attr, ed)
            grid.addWidget(ed, r, 1, 1, 4); r += 1

        hb = QHBoxLayout()
        hb.addWidget(QLabel("scroll = zoom · drag = pan · right-drag = box  ·  "
                            "click the plot then press A to autoscale"))
        grid.addLayout(hb, r, 0, 1, 5)
        v.addWidget(ga)

        # --- processing ---
        gp = QGroupBox("Signal processing")
        fp = QFormLayout(gp)
        self.chk_filter = QCheckBox("Low-pass filter")
        self.chk_filter.toggled.connect(self._processing_changed)
        self.spn_filter_hz = QDoubleSpinBox()
        self.spn_filter_hz.setDecimals(1)
        self.spn_filter_hz.setRange(1.0, 1e8)
        self.spn_filter_hz.setValue(10_000.0)
        self.spn_filter_hz.setSuffix(" Hz")
        self.spn_filter_hz.valueChanged.connect(self._processing_changed)
        self.cmb_filter_target = QComboBox()
        self.cmb_filter_target.addItems([
            "All enabled channels",
            "Left-axis channels",
            "Current-like channels",
        ])
        self.cmb_filter_target.currentIndexChanged.connect(
            self._processing_changed)
        fp.addRow(self.chk_filter)
        fp.addRow("Cutoff:", self.spn_filter_hz)
        fp.addRow("Apply to:", self.cmb_filter_target)
        v.addWidget(gp)

        # --- stats ---
        gs = QGroupBox("Statistics over visible window")
        vs = QVBoxLayout(gs)
        self.stats_table = QTableWidget(0, 6)
        self.stats_table.setHorizontalHeaderLabels(
            ["Channel", "Peak", "Min", "Mean", "RMS", "t@peak"])
        self.stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.stats_table.setMinimumHeight(120)
        btn_stats = QPushButton("Compute stats")
        btn_stats.clicked.connect(self.compute_stats)
        vs.addWidget(self.stats_table)
        vs.addWidget(btn_stats)
        v.addWidget(gs)

        # --- calibration ---
        gcal = QGroupBox("Reference calibration")
        fcal = QFormLayout(gcal)
        self.cmb_fit_source = QComboBox()
        self.cmb_fit_ref = QComboBox()
        fit_window = QWidget()
        fit_row = QHBoxLayout(fit_window)
        fit_row.setContentsMargins(0, 0, 0, 0)
        self.ed_fit_lo = QLineEdit()
        self.ed_fit_hi = QLineEdit()
        self.ed_fit_lo.setPlaceholderText("use visible min")
        self.ed_fit_hi.setPlaceholderText("use visible max")
        fit_row.addWidget(self.ed_fit_lo)
        fit_row.addWidget(QLabel("to"))
        fit_row.addWidget(self.ed_fit_hi)
        self.btn_fit_apply = QPushButton("Fit source -> ref and apply gain")
        self.btn_fit_apply.clicked.connect(self._fit_reference_gain)
        self.txt_fit = QPlainTextEdit()
        self.txt_fit.setReadOnly(True)
        self.txt_fit.setMaximumHeight(86)
        self.txt_fit.setPlaceholderText(
            "Uses a through-origin fit over the visible X window, or the "
            "custom window above, then multiplies the source-channel gain."
        )
        fcal.addRow("Source:", self.cmb_fit_source)
        fcal.addRow("Reference:", self.cmb_fit_ref)
        fcal.addRow("Fit window:", fit_window)
        fcal.addRow(self.btn_fit_apply)
        fcal.addRow(self.txt_fit)
        v.addWidget(gcal)

        # --- export ---
        ge = QGroupBox("Publication export (Nature style)")
        fe = QFormLayout(ge)
        self.cmb_width = QComboBox()
        self.cmb_width.addItems(["89 mm (single column)",
                                 "120 mm (1.5 column)",
                                 "183 mm (double column)"])
        self.spn_height = QDoubleSpinBox()
        self.spn_height.setRange(20, 250)
        self.spn_height.setValue(60)
        self.spn_height.setSuffix(" mm")
        self.spn_dpi = QSpinBox()
        self.spn_dpi.setRange(150, 1200)
        self.spn_dpi.setValue(600)
        self.chk_grid = QCheckBox("Dashed grid")
        self.chk_fullres = QCheckBox("Full resolution (slow for huge files)")
        fe.addRow("Width:", self.cmb_width)
        fe.addRow("Height:", self.spn_height)
        fe.addRow("JPG dpi:", self.spn_dpi)
        self.cmb_legend = QComboBox()
        self.cmb_legend.addItems(["best", "upper right", "upper left",
                                  "lower right", "lower left",
                                  "outside right"])
        fe.addRow("Legend:", self.cmb_legend)
        fe.addRow(self.chk_grid)
        fe.addRow(self.chk_fullres)
        btn_exp = QPushButton("Export SVG / JPG…")
        btn_exp.clicked.connect(self.export_pub)
        fe.addRow(btn_exp)
        v.addWidget(ge)

        v.addStretch(1)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidget(panel)
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumWidth(470)

    def _build_ai_panel(self):
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        title = QLabel("Local AI side chat")
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        v.addWidget(title)

        hai0 = QHBoxLayout()
        self.cmb_backend = QComboBox()
        self.cmb_backend.addItems(["MLX direct (Apple Silicon)",
                                   "Ollama (server)",
                                   "llama.cpp (Metal, GGUF file)"])
        self.cmb_backend.currentIndexChanged.connect(self._backend_changed)
        hai0.addWidget(QLabel("Backend:"))
        hai0.addWidget(self.cmb_backend, 1)
        v.addLayout(hai0)

        hai = QHBoxLayout()
        from ai_assistant import DEFAULT_GGUF
        self.ed_model = QLineEdit(DEFAULT_GGUF)
        self.btn_browse_gguf = QPushButton("Browse…")
        self.btn_browse_gguf.setToolTip(
            "MLX backend: choose a local MLX model folder. llama.cpp: "
            "choose a GGUF file.")
        self.btn_browse_gguf.clicked.connect(self._browse_gguf)
        hai.addWidget(QLabel("Model:"))
        hai.addWidget(self.ed_model, 1)
        hai.addWidget(self.btn_browse_gguf)
        v.addLayout(hai)

        # installed-model picker (Ollama backend): lists `ollama list`
        # models so users never have to remember tags
        inst_row = QHBoxLayout()
        self.cmb_installed = QComboBox()
        self.cmb_installed.setToolTip(
            "Models already downloaded in Ollama. Pick one to use it.")
        self.cmb_installed.activated.connect(self._installed_model_picked)
        self.btn_refresh_models = QPushButton("Refresh")
        self.btn_refresh_models.setToolTip(
            "Re-scan installed Ollama models (after `ollama pull …`).")
        self.btn_refresh_models.clicked.connect(self._refresh_installed_models)
        inst_row.addWidget(QLabel("Installed:"))
        inst_row.addWidget(self.cmb_installed, 1)
        inst_row.addWidget(self.btn_refresh_models)
        v.addLayout(inst_row)

        papers_row = QHBoxLayout()
        self.ed_papers = QLineEdit()
        self.ed_papers.setPlaceholderText("Optional folder with PDFs / notes")
        self.btn_browse_papers = QPushButton("Folder…")
        self.btn_browse_papers.clicked.connect(self._browse_papers)
        self.btn_index_papers = QPushButton("Index")
        self.btn_index_papers.setToolTip(
            "Index the PDFs/notes in the chosen folder for retrieval.")
        self.btn_index_papers.clicked.connect(self._index_papers)
        papers_row.addWidget(QLabel("Papers:"))
        papers_row.addWidget(self.ed_papers, 1)
        papers_row.addWidget(self.btn_browse_papers)
        papers_row.addWidget(self.btn_index_papers)
        v.addLayout(papers_row)

        self.lbl_papers = QLabel(
            "Paper retrieval is optional. It indexes local PDFs/text files "
            "for question-time context instead of trying to fine-tune a "
            "7B model on the MacBook."
        )
        self.lbl_papers.setWordWrap(True)
        v.addWidget(self.lbl_papers)

        self.txt_ai = QPlainTextEdit()
        self.txt_ai.setReadOnly(True)
        self.txt_ai.setPlaceholderText(
            "Ask about the visible data, compare channels, or request a "
            "sanity check. If you index a paper folder, relevant excerpts "
            "are added to the prompt automatically."
        )
        v.addWidget(self.txt_ai, 1)

        self.ed_ai_prompt = QPlainTextEdit()
        self.ed_ai_prompt.setMaximumHeight(96)
        self.ed_ai_prompt.setPlaceholderText(
            "Example: Compare the busbar and Pearson traces in the visible "
            "window, and tell me whether the Rogowski signal looks delayed."
        )
        v.addWidget(self.ed_ai_prompt)

        # 2x2 grid so labels never get clipped when the panel is narrow
        ai_btns = QGridLayout()
        ai_btns.setContentsMargins(0, 0, 0, 0)
        self.btn_ai_summary = QPushButton("Summarize visible")
        self.btn_ai_summary.setToolTip(
            "Ask the model to summarize the visible data.")
        self.btn_ai_summary.clicked.connect(self.run_ai)
        self.btn_ai_anomaly = QPushButton("Detect anomalies")
        self.btn_ai_anomaly.setToolTip(
            "Deterministic NumPy scan of the visible window (no LLM).")
        self.btn_ai_anomaly.clicked.connect(self.run_anomaly_scan)
        self.btn_ai_send = QPushButton("Send")
        self.btn_ai_send.setDefault(True)
        self.btn_ai_send.clicked.connect(self._send_ai_prompt)
        self.btn_ai_clear = QPushButton("Clear chat")
        self.btn_ai_clear.clicked.connect(self._clear_chat)
        for b in (self.btn_ai_summary, self.btn_ai_anomaly,
                  self.btn_ai_send, self.btn_ai_clear):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_ai_saturation = QPushButton("Estimate saturation")
        self.btn_ai_saturation.setToolTip(
            "Deterministic fit: detects a clipped/saturated run, projects "
            "the post-saturation droop slope back to estimate the true "
            "peak (95% CI), and cross-calibrates against the second "
            "visible channel. No LLM involved.")
        self.btn_ai_saturation.clicked.connect(self.run_saturation_estimate)
        self.btn_ai_saturation.setSizePolicy(QSizePolicy.Expanding,
                                             QSizePolicy.Fixed)
        self.btn_ai_recon = QPushButton("Recover hidden peak")
        self.btn_ai_recon.setToolTip(
            "One-click saturated-pulse recovery: infers the BBCM/busbar "
            "channel, applies the calibrated display conversion, uses the "
            "rig saturation/reference defaults, estimates the hidden peak, "
            "then draws the censored-RLC reconstruction with a 95% band.")
        self.btn_ai_recon.clicked.connect(self.run_hidden_peak_recovery)
        self.btn_ai_recon.setSizePolicy(QSizePolicy.Expanding,
                                        QSizePolicy.Fixed)
        ai_btns.addWidget(self.btn_ai_summary, 0, 0)
        ai_btns.addWidget(self.btn_ai_anomaly, 0, 1)
        ai_btns.addWidget(self.btn_ai_saturation, 1, 0)
        ai_btns.addWidget(self.btn_ai_recon, 1, 1)
        ai_btns.addWidget(self.btn_ai_send, 2, 0)
        ai_btns.addWidget(self.btn_ai_clear, 2, 1)
        v.addLayout(ai_btns)

        self.btn_pipeline = QPushButton("▶  Analyze shot (full pipeline)")
        self.btn_pipeline.setToolTip(
            "One click, deterministic, in order: zero baselines -> stats "
            "-> anomaly scan -> saturation estimate (rig defaults from "
            "USER_DEFAULTS) -> RLC reconstruction with the Pearson "
            "constraint. Results land in the chat; ask follow-ups there.")
        self.btn_pipeline.setStyleSheet("font-weight: 600;")
        self.btn_pipeline.clicked.connect(self.run_shot_pipeline)
        v.addWidget(self.btn_pipeline)

        ovl_row = QHBoxLayout()
        self.chk_sat_overlay = QCheckBox("Saturation fit")
        self.chk_sat_overlay.setChecked(True)
        self.chk_sat_overlay.setToolTip(
            "Overlay the rise/droop projections and reconstructed peak "
            "from the last saturation estimate onto the main plot.")
        self.chk_sat_overlay.toggled.connect(
            lambda _on: self.apply_sat_overlay())
        self.chk_recon_overlay = QCheckBox("RLC reconstruction")
        self.chk_recon_overlay.setChecked(True)
        self.chk_recon_overlay.setToolTip(
            "Overlay the censored-ML RLC reconstruction (curve + 95% "
            "band) from the last reconstruct run for visual comparison "
            "with the measured traces.")
        self.chk_recon_overlay.toggled.connect(
            lambda _on: self.apply_recon_overlay())
        ovl_row.addWidget(QLabel("Overlays:"))
        ovl_row.addWidget(self.chk_sat_overlay)
        ovl_row.addWidget(self.chk_recon_overlay)
        ovl_row.addStretch(1)
        v.addLayout(ovl_row)

        self.ai_panel = panel
        self.ai_panel.setMinimumWidth(380)
        self._backend_changed(0)

    # ----------------------------------------------------------- file ------
    def open_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open oscilloscope CSV", "",
            "Data files (*.csv *.txt *.tsv);;All files (*)")
        if not path:
            return
        self.open_csv_path(path)

    def load_example_shot(self):
        """Open the bundled synthetic shot so a first-time user can explore
        every feature without having their own scope file yet."""
        here = os.path.dirname(os.path.abspath(__file__))
        sample = os.path.join(here, "examples", "synthetic_vi_didt_scope.csv")
        if not os.path.exists(sample):
            QMessageBox.information(
                self, "Example shot",
                "Bundled example not found. Generate it with:\n"
                "    python scripts/generate_synthetic_vi_didt.py")
            return
        self.open_csv_path(sample)

    def open_csv_path(self, path: str):
        """Load a CSV through the one approved path.

        The source file is hashed and kept read-only; all formulas, filters,
        reconstructions, and overlays stay in RAM/display state.
        """
        self._sat_overlay = None          # stale fits don't apply to a
        self._recon_overlay = None        # newly loaded file
        self.clear_overlay_shots()
        try:
            session = DataSession.from_path(path)
            data = load_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return
        self.session = session
        self.data = data
        self.data_quality = quality_report(data)
        sidecar_note = ""
        try:
            sidecar_path, metadata = ensure_sidecar(
                data, session, self.data_quality)
            self.shot_metadata = metadata
            self.shot_metadata_path = str(sidecar_path)
            sidecar_note = f" Metadata sidecar: {sidecar_path.name}."
        except Exception as exc:
            self.shot_metadata = None
            self.shot_metadata_path = None
            sidecar_note = f" Metadata sidecar failed: {exc}."
        model = (self.data.meta.get("Model") or [""])[0]
        self.lbl_file.setText(
            f"{os.path.basename(path)}"
            + (f" — {model}" if model else "")
            + f" — {self.data.n_rows:,} rows × {len(self.data.columns)} cols"
            + f" — {self.data_quality.status.upper()}")
        self._undo_stack.clear()
        self._transform_cache.clear()
        self._init_channels()
        self.refresh_plot()
        self._refresh_embedded_3d()
        self.pi.vb.autoRange()
        self.statusBar().showMessage(
            "CSV loaded read-only; original file hash recorded. "
            + self.data_quality.one_line()
            + sidecar_note, 9000)

    def _init_channels(self):
        cols = self.data.columns
        self.cmb_x.blockSignals(True)
        self.cmb_x.clear()
        self.cmb_x.addItems(cols)
        # heuristics: pick a time-ish column for X
        xi = next((i for i, c in enumerate(cols)
                   if "time" in c.lower() or c.lower() in ("t", "x", "s")), 0)
        self.cmb_x.setCurrentIndex(xi)
        self.cmb_x.blockSignals(False)

        self.channels = []
        n_enabled = 0
        for i, c in enumerate(cols):
            if i == xi:
                continue
            unit = self.data.units.get(c, "")
            is_aux = "peak detect" in c.lower()   # Tek companion columns
            en = (not is_aux) and n_enabled < 4
            n_enabled += int(en)
            self.channels.append(Channel(
                name=c, enabled=en,
                unit=unit,
                color="#000000"))
        self._apply_palette_to_channels(rebuild=False)
        self._rebuild_table()
        self._refresh_fit_combos()

    def _rebuild_table(self):
        self._building_table = True
        t = self.table
        t.setRowCount(len(self.channels))
        for r, ch in enumerate(self.channels):
            cb = QCheckBox()
            cb.setChecked(ch.enabled)
            cb.toggled.connect(self._table_edited)
            w = QWidget(); l = QHBoxLayout(w)
            l.setContentsMargins(8, 0, 0, 0); l.addWidget(cb)
            t.setCellWidget(r, 0, w)

            it = QTableWidgetItem(ch.name)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            it.setForeground(QColor(ch.color))
            t.setItem(r, 1, it)

            ax = QComboBox(); ax.addItems(["Left", "Right"])
            ax.setCurrentIndex(0 if ch.axis == "left" else 1)
            ax.currentIndexChanged.connect(self._table_edited)
            t.setCellWidget(r, 2, ax)

            pr = QComboBox(); pr.addItems(list(self.presets))
            if ch.preset_name in self.presets:
                pr.setCurrentText(ch.preset_name)
            pr.currentTextChanged.connect(
                lambda name, row=r: self._apply_preset(row, name))
            t.setCellWidget(r, 3, pr)

            t.setItem(r, 4, QTableWidgetItem(f"{ch.gain:g}"))
            t.setItem(r, 5, QTableWidgetItem(f"{ch.offset:g}"))
            t.setItem(r, 6, QTableWidgetItem(ch.formula))
            t.setItem(r, 7, QTableWidgetItem(ch.label))

            hdr = QTableWidgetItem("■")
            hdr.setForeground(QColor(ch.color))
            t.setVerticalHeaderItem(r, hdr)
        self._building_table = False

    def _apply_preset(self, row: int, name: str):
        if self._building_table or name not in self.presets:
            return
        p = self.presets[name]
        self._building_table = True
        self.channels[row].preset_name = name
        self.channels[row].unit = p.get("unit", "")
        self.table.item(row, 4).setText(f"{p['gain']:g}")
        self.table.item(row, 5).setText(f"{p['offset']:g}")
        self.table.item(row, 6).setText(p.get("formula", "x"))
        self._building_table = False
        self._table_edited()

    def _save_presets(self):
        path = os.path.join(HERE, "presets.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.presets, f, indent=2)
            f.write("\n")

    def _selected_channel_row(self) -> int:
        row = self.table.currentRow()
        if 0 <= row < len(self.channels):
            return row
        selected = self.table.selectionModel().selectedRows()
        if selected:
            row = selected[0].row()
            if 0 <= row < len(self.channels):
                return row
        for item in self.table.selectedItems():
            row = item.row()
            if 0 <= row < len(self.channels):
                return row
        return -1

    def _validate_formula_for_channel(self, ch: Channel, formula: str):
        if self.data is None:
            return
        raw_x = self._raw_x()
        if raw_x is None or ch.name not in self.data.df:
            return
        raw_y = self.data.df[ch.name].to_numpy(dtype=np.float64)
        n = min(len(raw_x), len(raw_y), 5000)
        if n < 2:
            return
        evaluate_formula(formula, raw_y[:n], raw_x[:n], raw_x[:n] * 1000.0)

    def _open_formula_dialog(self):
        row = self._selected_channel_row()
        if row < 0:
            QMessageBox.information(
                self, "Formula editor",
                "Select a channel row first, then open the formula editor."
            )
            return

        ch = self.channels[row]
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Formula editor - {ch.name}")
        outer = QVBoxLayout(dlg)

        form = QFormLayout()
        ed_preset = QLineEdit()
        ed_preset.setPlaceholderText("Optional new preset name")
        ed_unit = QLineEdit(ch.unit)
        ed_gain = QLineEdit(f"{ch.gain:g}")
        ed_offset = QLineEdit(f"{ch.offset:g}")
        ed_label = QLineEdit(ch.label)
        txt_formula = QPlainTextEdit(ch.formula)
        txt_formula.setMinimumWidth(560)
        txt_formula.setMinimumHeight(140)

        form.addRow("Save as preset:", ed_preset)
        form.addRow("Unit:", ed_unit)
        form.addRow("Gain:", ed_gain)
        form.addRow("Offset:", ed_offset)
        form.addRow("Label:", ed_label)
        form.addRow("Formula:", txt_formula)
        outer.addLayout(form)

        help_text = QLabel(
            FORMULA_HINT + "\nExample: "
            "`baseline(lowpass((x - 2.5) * 1500 / 2, t, 1.5e4), "
            "t_ms, -1.0)`"
        )
        help_text.setWordWrap(True)
        outer.addWidget(help_text)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Apply / Save"
        )
        outer.addWidget(buttons)

        def apply_and_close():
            formula = txt_formula.toPlainText().strip() or "x"
            try:
                gain = float(ed_gain.text().strip() or "1")
                offset = float(ed_offset.text().strip() or "0")
            except ValueError:
                QMessageBox.warning(
                    dlg, "Formula editor",
                    "Gain and offset must be numeric."
                )
                return

            try:
                self._validate_formula_for_channel(ch, formula)
            except FormulaError as exc:
                QMessageBox.warning(
                    dlg, "Formula editor",
                    f"Formula did not validate for this channel:\n{exc}"
                )
                return

            ch.unit = ed_unit.text().strip()
            ch.gain = gain
            ch.offset = offset
            ch.label = ed_label.text()
            ch.formula = formula

            preset_name = ed_preset.text().strip()
            if preset_name:
                self.presets[preset_name] = {
                    "gain": gain,
                    "offset": offset,
                    "unit": ch.unit,
                    "formula": formula,
                }
                ch.preset_name = preset_name
                try:
                    self._save_presets()
                except Exception as exc:
                    QMessageBox.warning(
                        dlg, "Formula editor",
                        f"Could not save presets.json:\n{exc}"
                    )
                    return

            self._transform_cache.clear()
            self._rebuild_table()
            self.table.selectRow(row)
            self._refresh_fit_combos()
            self.refresh_plot()
            dlg.accept()

        buttons.accepted.connect(apply_and_close)
        buttons.rejected.connect(dlg.reject)
        dlg.exec()

    def _pick_color(self, row: int):
        if row >= len(self.channels):
            return
        c = QColorDialog.getColor(QColor(self.channels[row].color), self)
        if c.isValid():
            self.channels[row].color = c.name()
            self._rebuild_table()
            self.refresh_plot()

    def _table_edited(self, *_):
        if self._building_table or not self.channels:
            return
        self._transform_cache.clear()
        for r, ch in enumerate(self.channels):
            w = self.table.cellWidget(r, 0)
            ch.enabled = w.findChild(QCheckBox).isChecked() if w else False
            axw = self.table.cellWidget(r, 2)
            ch.axis = "left" if (axw and axw.currentIndex() == 0) else "right"
            prw = self.table.cellWidget(r, 3)
            if prw:
                ch.preset_name = prw.currentText()
            try:
                ch.gain = float(self.table.item(r, 4).text())
            except (TypeError, ValueError):
                pass
            try:
                ch.offset = float(self.table.item(r, 5).text())
            except (TypeError, ValueError):
                pass
            it = self.table.item(r, 6)
            ch.formula = (it.text() if it else "x").strip() or "x"
            it = self.table.item(r, 7)
            ch.label = it.text() if it else ""
        self._refresh_fit_combos()
        self.refresh_plot()

    def _current_palette(self) -> list[str]:
        return PALETTES.get(self.cmb_palette.currentText(),
                            next(iter(PALETTES.values())))

    def _apply_palette_to_channels(self, *_ , rebuild: bool = True):
        palette = self._current_palette()
        for i, ch in enumerate(self.channels):
            ch.color = palette[i % len(palette)]
        if rebuild and self.channels:
            self._rebuild_table()
            self.refresh_plot()

    def _refresh_fit_combos(self):
        combos = [self.cmb_fit_source, self.cmb_fit_ref]
        current = [cb.currentData() for cb in combos]
        for idx, cb in enumerate(combos):
            cb.blockSignals(True)
            cb.clear()
            for ch in self.channels:
                cb.addItem(ch.display_label(), ch.name)
            if current[idx]:
                pos = cb.findData(current[idx])
                if pos >= 0:
                    cb.setCurrentIndex(pos)
            cb.blockSignals(False)
        if self.cmb_fit_ref.count() > 1 and self.cmb_fit_ref.currentIndex() == 0:
            self.cmb_fit_ref.setCurrentIndex(1)

    def _x_column_changed(self, *_):
        self._transform_cache.clear()
        self.refresh_plot()

    def _processing_changed(self, *_):
        self._transform_cache.clear()
        self.refresh_plot()

    def _raw_x(self) -> np.ndarray | None:
        if self.data is None or self.cmb_x.currentIndex() < 0:
            return None
        return self.data.df[self.cmb_x.currentText()].to_numpy(dtype=np.float64)

    # ----------------------------------------------------------- plot ------
    def _x(self) -> np.ndarray | None:
        raw_x = self._raw_x()
        if raw_x is None:
            return None
        return raw_x * self.spn_xscale.value()

    def _channel_by_name(self, name: str) -> Channel | None:
        return next((ch for ch in self.channels if ch.name == name), None)

    def _channel_data(self, ch: Channel) -> np.ndarray:
        raw_x = self._raw_x()
        if self.data is None or raw_x is None:
            raise FormulaError("Load a CSV first.")
        key = (ch.name, self.cmb_x.currentText(), ch.formula,
               float(ch.gain), float(ch.offset),
               self._processing_signature(ch))
        cached = self._transform_cache.get(key)
        if cached is not None:
            return cached
        raw_y = self.data.df[ch.name].to_numpy(dtype=np.float64)
        y = ch.convert(raw_y, raw_x, raw_x * 1000.0)
        if self._filter_applies_to(ch):
            y = lowpass(y, raw_x, self.spn_filter_hz.value())
        self._transform_cache[key] = y
        return y

    def _processing_signature(self, ch: Channel) -> tuple:
        if not getattr(self, "chk_filter", None) or not self.chk_filter.isChecked():
            return ("filter", False)
        return (
            "filter",
            True,
            round(float(self.spn_filter_hz.value()), 6),
            self.cmb_filter_target.currentIndex(),
            self._filter_applies_to(ch),
        )

    def _filter_applies_to(self, ch: Channel) -> bool:
        if not getattr(self, "chk_filter", None) or not self.chk_filter.isChecked():
            return False
        target = self.cmb_filter_target.currentText()
        if target == "All enabled channels":
            return True
        if target == "Left-axis channels":
            return ch.axis == "left"
        label = ch.display_label().lower()
        return "(a)" in label or "current" in label or ch.unit.lower() == "a"

    # ---------------------------------------------------- reversible views --
    def _capture_display_state(self) -> dict:
        xr, yl = self.pi.vb.viewRange()
        yr = self.vb_right.viewRange()[1]
        return {
            "xscale": float(self.spn_xscale.value()),
            "filter_on": bool(self.chk_filter.isChecked()),
            "filter_hz": float(self.spn_filter_hz.value()),
            "filter_target": self.cmb_filter_target.currentText(),
            "palette": self.cmb_palette.currentText(),
            "xlabel": self.ed_xlabel.text(),
            "ylabel_left": self.ed_yllabel.text(),
            "ylabel_right": self.ed_yrlabel.text(),
            "title": self.ed_title.text(),
            "top_on": bool(self.chk_top.isChecked()),
            "top_scale": float(self.spn_topscale.value()),
            "top_label": self.ed_toplabel.text(),
            "zero_align": bool(self.chk_zero.isChecked()),
            "xrange": tuple(float(v) for v in xr),
            "yrange_left": tuple(float(v) for v in yl),
            "yrange_right": tuple(float(v) for v in yr),
            "sat_overlay": getattr(self, "_sat_overlay", None),
            "recon_overlay": getattr(self, "_recon_overlay", None),
            "channels": [
                {
                    "name": ch.name,
                    "enabled": ch.enabled,
                    "axis": ch.axis,
                    "gain": ch.gain,
                    "offset": ch.offset,
                    "label": ch.label,
                    "color": ch.color,
                    "formula": ch.formula,
                    "unit": ch.unit,
                    "preset_name": ch.preset_name,
                }
                for ch in self.channels
            ],
        }

    def push_display_undo(self, reason: str):
        if self.data is None:
            return
        self._undo_stack.append((reason, self._capture_display_state()))
        if len(self._undo_stack) > 25:
            self._undo_stack.pop(0)

    def _restore_display_state(self, state: dict):
        self.spn_xscale.setValue(float(state.get("xscale", 1000.0)))
        self.chk_filter.setChecked(bool(state.get("filter_on", False)))
        self.spn_filter_hz.setValue(float(state.get("filter_hz", 10_000.0)))
        idx = self.cmb_filter_target.findText(
            state.get("filter_target", "All enabled channels"))
        if idx >= 0:
            self.cmb_filter_target.setCurrentIndex(idx)
        idx = self.cmb_palette.findText(state.get("palette", "Wong / Nature"))
        if idx >= 0:
            self.cmb_palette.setCurrentIndex(idx)
        for attr, key in [
                ("ed_xlabel", "xlabel"),
                ("ed_yllabel", "ylabel_left"),
                ("ed_yrlabel", "ylabel_right"),
                ("ed_title", "title"),
                ("ed_toplabel", "top_label")]:
            getattr(self, attr).setText(str(state.get(key, "")))
        self.chk_top.setChecked(bool(state.get("top_on", False)))
        self.spn_topscale.setValue(float(state.get("top_scale", 1.0)))
        self.chk_zero.setChecked(bool(state.get("zero_align", True)))
        by_name = {ch.name: ch for ch in self.channels}
        for spec in state.get("channels", []):
            ch = by_name.get(spec.get("name"))
            if ch is None:
                continue
            for key in ("enabled", "axis", "gain", "offset", "label",
                        "color", "formula", "unit", "preset_name"):
                if key in spec:
                    setattr(ch, key, spec[key])
        self._sat_overlay = state.get("sat_overlay")
        self._recon_overlay = state.get("recon_overlay")
        self._transform_cache.clear()
        self._rebuild_table()
        self.refresh_plot()
        try:
            x0, x1 = state.get("xrange", self.pi.vb.viewRange()[0])
            y0, y1 = state.get("yrange_left", self.pi.vb.viewRange()[1])
            r0, r1 = state.get("yrange_right", self.vb_right.viewRange()[1])
            self.pi.vb.setXRange(x0, x1, padding=0)
            self.pi.vb.setYRange(y0, y1, padding=0)
            self.vb_right.setYRange(r0, r1, padding=0)
        except Exception:
            pass
        self.apply_sat_overlay()
        self.apply_recon_overlay()

    def undo_last_ai_change(self):
        if not self._undo_stack:
            self.statusBar().showMessage("Nothing to undo.", 4000)
            return
        reason, state = self._undo_stack.pop()
        self._restore_display_state(state)
        self._append_chat("System", f"Undid display change: {reason}")
        self.statusBar().showMessage(
            "Previous display/session state restored. CSV untouched.", 7000)

    def reset_to_raw_view(self):
        if self.data is None:
            return
        self.push_display_undo("reset to raw view")
        self.chk_filter.setChecked(False)
        self._sat_overlay = None
        self._recon_overlay = None
        self.apply_sat_overlay()
        self.apply_recon_overlay()
        self.clear_overlay_shots()
        self._transform_cache.clear()
        self.refresh_plot()
        self.statusBar().showMessage(
            "Display reset; original CSV was not modified.", 7000)

    def refresh_plot(self):
        if self.data is None:
            return
        x = self._x()
        for item in list(self.vb_right.addedItems):
            self.vb_right.removeItem(item)
        self.pi.clear()
        self.curves.clear()
        legend = self.pi.addLegend(offset=(10, 10), labelTextSize="9pt",
                                   brush=pg.mkBrush(255, 255, 255, 220),
                                   pen=pg.mkPen("#444444", width=0.5))
        legend.clear()

        any_right = False
        for ch in self.channels:
            if not ch.enabled:
                continue
            try:
                y = self._channel_data(ch)
            except FormulaError as exc:
                self.statusBar().showMessage(
                    f"{ch.name}: formula error - {exc}", 9000
                )
                continue
            except Exception as exc:
                self.statusBar().showMessage(
                    f"{ch.name}: transform failed - {exc}", 9000
                )
                continue
            # decimate=False: pyqtgraph's own auto peak-downsampling
            # (enabled inside make_curve) handles the main scope view so
            # zooming in still reaches full resolution. Pre-decimation is
            # for paths without dynamic downsampling (e.g. overlay shots).
            curve = make_curve(x, y, ch.color, ch.display_label(),
                               decimate=False)
            if ch.axis == "right":
                any_right = True
                self.vb_right.addItem(curve)
                # legend entry for right-axis curves
                self.pi.legend.addItem(curve, ch.display_label())
            else:
                self.pi.addItem(curve)
            # view-dependent settings MUST come after addItem (pyqtgraph
            # needs a parent view first), else the trace clips to nothing
            # and autorange runs away — the blank-plot regression.
            enable_view_downsampling(curve)
            self.curves[ch.name] = curve

        self.right_axis.setVisible(any_right)
        self.right_axis.setLabel(self.ed_yrlabel.text(), color="#D55E00")
        self.right_axis.setPen(pg.mkPen("#D55E00"))
        self.right_axis.setTextPen(pg.mkPen("#D55E00"))
        self.pi.setLabel("bottom", self.ed_xlabel.text())
        self.pi.setLabel("left", self.ed_yllabel.text())
        self.pi.setTitle(self.ed_title.text() or None,
                         color="k", size="11pt")

        self.top_axis.scale_f = self.spn_topscale.value()
        self.top_axis.setVisible(self.chk_top.isChecked())
        if self.chk_top.isChecked():
            self.top_axis.setLabel(self.ed_toplabel.text())

        self._sync_right_geometry()
        if any_right:
            self.vb_right.enableAutoRange(y=True)
        self._maybe_align_zero()

    def _maybe_align_zero(self, *_):
        if not self.chk_zero.isChecked():
            return
        if not any(c.enabled and c.axis == "right" for c in self.channels):
            return
        (l0, l1) = self.pi.vb.viewRange()[1]
        if l1 <= l0:
            return
        frac = min(max((0.0 - l0) / (l1 - l0), 1e-3), 1 - 1e-3)
        (r0, r1) = self.vb_right.viewRange()[1]
        span = 0.0
        if r1 > 0:
            span = max(span, r1 / (1 - frac))
        if r0 < 0:
            span = max(span, -r0 / frac)
        if span <= 0:
            return
        self.vb_right.blockSignals(True)
        self.vb_right.setYRange(-frac * span, (1 - frac) * span, padding=0)
        self.vb_right.blockSignals(False)

    def _set_range(self, which: str):
        def f(ed):
            try:
                return float(ed.text())
            except ValueError:
                return None
        if which == "x":
            lo, hi = f(self.x_lo), f(self.x_hi)
            if lo is not None and hi is not None:
                self.pi.vb.setXRange(lo, hi, padding=0)
        elif which == "yl":
            lo, hi = f(self.yl_lo), f(self.yl_hi)
            if lo is not None and hi is not None:
                self.pi.vb.setYRange(lo, hi, padding=0)
        else:
            lo, hi = f(self.yr_lo), f(self.yr_hi)
            if lo is not None and hi is not None:
                self.chk_zero.setChecked(False)
                self.vb_right.setYRange(lo, hi, padding=0)

    def _reset_zoom(self):
        self.pi.vb.autoRange()
        self.vb_right.enableAutoRange(y=True)

    def _fit_reference_gain(self):
        if self.data is None:
            self.txt_fit.setPlainText("Load a file first.")
            return
        src = self._channel_by_name(self.cmb_fit_source.currentData())
        ref = self._channel_by_name(self.cmb_fit_ref.currentData())
        if not src or not ref or src.name == ref.name:
            self.txt_fit.setPlainText("Choose two different channels.")
            return
        x = self._x()
        try:
            y_src = self._channel_data(src)
            y_ref = self._channel_data(ref)
        except Exception as exc:
            self.txt_fit.setPlainText(f"Cannot fit: {exc}")
            return
        try:
            lo = float(self.ed_fit_lo.text()) if self.ed_fit_lo.text() else None
            hi = float(self.ed_fit_hi.text()) if self.ed_fit_hi.text() else None
        except ValueError:
            self.txt_fit.setPlainText("Fit window bounds must be numeric.")
            return
        if lo is None or hi is None:
            lo, hi = self.pi.vb.viewRange()[0]
        try:
            result = fit_forced_origin_gain(x, y_src, y_ref, lo, hi)
        except CalibrationError as exc:
            self.txt_fit.setPlainText(str(exc))
            return
        src.gain *= result.slope
        self.txt_fit.setPlainText(
            format_gain_fit(result, src.display_label(), self.ed_xlabel.text())
        )
        self._rebuild_table()
        self._transform_cache.clear()
        self.refresh_plot()

    # ---------------------------------------------------------- stats ------
    def compute_stats(self) -> str:
        if self.data is None:
            return ""
        x = self._x()
        (x0, x1) = self.pi.vb.viewRange()[0]
        m = (x >= x0) & (x <= x1)
        rows, lines = [], []
        for ch in self.channels:
            if not ch.enabled:
                continue
            try:
                y = self._channel_data(ch)[m]
            except Exception as exc:
                self.statusBar().showMessage(
                    f"Stats skipped for {ch.name}: {exc}", 7000
                )
                continue
            xv = x[m]
            if y.size == 0:
                continue
            pk, mn = float(np.nanmax(y)), float(np.nanmin(y))
            mean = float(np.nanmean(y))
            rms = float(np.sqrt(np.nanmean(np.square(y, dtype=np.float64))))
            tpk = float(xv[int(np.nanargmax(y))])
            rows.append((ch.display_label(), pk, mn, mean, rms, tpk))
            lines.append(f"{ch.display_label()}: peak={pk:.4g}, min={mn:.4g}, "
                         f"mean={mean:.4g}, RMS={rms:.4g}, "
                         f"pk-pk={pk - mn:.4g}, t@peak={tpk:.4g}")
        self.stats_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                txt = val if isinstance(val, str) else f"{val:.4g}"
                self.stats_table.setItem(r, c, QTableWidgetItem(txt))
        header = (f"Window: {x0:.4g} → {x1:.4g} "
                  f"({self.ed_xlabel.text()}), {int(m.sum()):,} samples")
        return header + "\n" + "\n".join(lines)

    # ------------------------------------------------------------- AI ------
    def _backend_changed(self, idx: int):
        if idx == 0:   # MLX direct
            from ai_assistant import DEFAULT_MLX_MODEL
            self.ed_model.setText(DEFAULT_MLX_MODEL)
    
            self.btn_browse_gguf.setVisible(True)
            self.btn_browse_gguf.setText("Folder…")
            self.btn_browse_gguf.setToolTip(
                "Choose an MLX model folder. Scope Analyzer scans plugged-in "
                "model vaults first, then ~/models/mlx."
            )
    
            self.cmb_installed.setEnabled(True)
            self.btn_refresh_models.setEnabled(True)
            self.cmb_installed.setToolTip(
                "Complete MLX model folders found on the model vault drive "
                "or in ~/models/mlx."
            )
            self._refresh_installed_models()
    
        elif idx == 1:  # Ollama
            from ai_assistant import DEFAULT_OLLAMA_MODEL
            self.ed_model.setText(DEFAULT_OLLAMA_MODEL)
    
            self.btn_browse_gguf.setVisible(False)
    
            self.cmb_installed.setEnabled(True)
            self.btn_refresh_models.setEnabled(True)
            self.cmb_installed.setToolTip(
                "Models already downloaded in Ollama."
            )
            self._refresh_installed_models()
    
        else:  # llama.cpp
            from ai_assistant import DEFAULT_GGUF
            self.ed_model.setText(DEFAULT_GGUF)
    
            self.btn_browse_gguf.setVisible(True)
            self.btn_browse_gguf.setText("Browse…")
            self.btn_browse_gguf.setToolTip(
                "Choose a GGUF file for llama.cpp."
            )
    
            self.cmb_installed.clear()
            self.cmb_installed.addItem("(Use Browse… to choose a GGUF file)")
            self.cmb_installed.setEnabled(False)
            self.btn_refresh_models.setEnabled(False)
    
    
    def _refresh_installed_models(self):
        self.cmb_installed.clear()
        idx = self.cmb_backend.currentIndex()
    
        if idx == 0:  # MLX direct
            from ai_assistant import list_mlx_models
            names = list_mlx_models()
    
            if names:
                # Display shorter labels but store full paths as item data.
                for path in names:
                    label = os.path.basename(path.rstrip(os.sep))
                    self.cmb_installed.addItem(label, path)
    
                cur = os.path.abspath(os.path.expanduser(
                    self.ed_model.text().strip()
                ))
                for i in range(self.cmb_installed.count()):
                    if self.cmb_installed.itemData(i) == cur:
                        self.cmb_installed.setCurrentIndex(i)
                        break
            else:
                self.cmb_installed.addItem(
                    "(No local MLX folders; plug in model drive or use HF id)"
                )
    
        elif idx == 1:  # Ollama
            from ai_assistant import list_ollama_models
            names = list_ollama_models()
    
            if names:
                self.cmb_installed.addItems(names)
                cur = self.ed_model.text().strip()
                if cur in names:
                    self.cmb_installed.setCurrentIndex(names.index(cur))
            else:
                self.cmb_installed.addItem("(Ollama not running / no models)")
    
        else:  # llama.cpp
            self.cmb_installed.addItem("(Use Browse… to choose a GGUF file)")
    
    
    def _installed_model_picked(self, idx: int):
        if idx < 0:
            return
    
        # For MLX direct, itemData stores the full folder path.
        data = self.cmb_installed.itemData(idx)
        if data:
            self.ed_model.setText(str(data))
            return
    
        # For Ollama, the visible text is the model tag.
        name = self.cmb_installed.itemText(idx)
        if name and not name.startswith("("):
            self.ed_model.setText(name)

    def _browse_gguf(self):
        if self.cmb_backend.currentIndex() == 0:
            from ai_assistant import mlx_model_roots
            roots = [root for root in mlx_model_roots() if os.path.isdir(root)]
            default_dir = roots[0] if roots else os.path.expanduser("~/models")
            path = QFileDialog.getExistingDirectory(
                self, "Choose local MLX model folder",
                default_dir)
            if path:
                from ai_assistant import resolve_mlx_model
                try:
                    resolved = resolve_mlx_model(path)
                except ValueError as exc:
                    self.ed_model.setText(path)
                    self.statusBar().showMessage(
                        f"MLX model folder not usable: {exc}", 12000)
                else:
                    self.ed_model.setText(resolved)
                    self.statusBar().showMessage(
                        f"MLX model selected: "
                        f"{os.path.basename(resolved.rstrip(os.sep))}",
                        7000)
                    self._refresh_installed_models()
            return
        default_dir = "/Volumes/ScopeStudioModels/gguf"
        if not os.path.isdir(default_dir):
            default_dir = os.path.expanduser("~/models")
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose GGUF model", default_dir,
            "GGUF models (*.gguf);;All files (*)")
        if path:
            self.ed_model.setText(path)

    def _browse_papers(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose paper folder", os.path.expanduser("~"))
        if path:
            self.ed_papers.setText(path)

    def create_draft_tool_template(self):
        from tool_sandbox import create_draft_tool
        paths = create_draft_tool(
            name="assistant_analysis_tool",
            purpose=(
                "Draft deterministic analysis helper. Review before "
                "promotion; original CSV files must remain untouched."
            ),
        )
        self._append_chat(
            "System",
            "Draft tool template created but NOT approved:\n"
            f"{paths.folder}\n"
            "Run its test and review the manifest before promotion."
        )
        QDesktopServices.openUrl(QUrl.fromLocalFile(paths.folder))

    def open_tool_sandbox(self):
        from tool_sandbox import ensure_sandbox
        QDesktopServices.openUrl(QUrl.fromLocalFile(ensure_sandbox()))

    def _index_papers(self):
        folder = self.ed_papers.text().strip()
        if not folder:
            self.lbl_papers.setText("Choose a folder first.")
            return
        self.lbl_papers.setText("Indexing papers locally…")
        self.btn_index_papers.setEnabled(False)
        self._paper_worker = PaperIndexWorker(folder)
        self._paper_worker.done.connect(self._papers_indexed)
        self._paper_worker.start()

    def _papers_indexed(self, index, message: str):
        self._paper_index = index
        self.lbl_papers.setText(message)
        self.btn_index_papers.setEnabled(True)

    def _append_chat(self, role: str, text: str):
        self.txt_ai.appendPlainText(f"{role}:\n{text}\n")
        cursor = self.txt_ai.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.txt_ai.setTextCursor(cursor)
        self.txt_ai.ensureCursorVisible()

    def _clear_chat(self):
        self.txt_ai.clear()
        self.ed_ai_prompt.clear()
        self._chat_history.clear()
        self._ai_trace_events.clear()

    def _build_visible_context(self) -> str:
        stats = self.compute_stats()
        if not stats:
            return ""
        x = self._x()
        (x0, x1) = self.pi.vb.viewRange()[0]
        mask = (x >= x0) & (x <= x1)
        lines = [
            f"Visible data window on screen: {x0:.4g} to {x1:.4g} "
            f"({self.ed_xlabel.text()})",
            f"Left axis label: {self.ed_yllabel.text()}",
            f"Right axis label: {self.ed_yrlabel.text()}",
            "Data policy: original CSV is read-only; all listed formulas, "
            "filters, reconstructions, and overlays are display/session "
            "state in RAM unless explicitly exported as a new file.",
            "",
            "Active display transforms:",
            self._active_transform_context(),
            "",
            stats,
            "",
            "Waveform previews (decimated visible points):",
        ]
        for ch in self.channels:
            if not ch.enabled:
                continue
            try:
                y = self._channel_data(ch)[mask]
            except Exception as exc:
                lines.append(f"- {ch.display_label()}: preview unavailable ({exc})")
                continue
            xv = x[mask]
            if y.size == 0:
                continue
            xs, ys = minmax_decimate(np.asarray(xv), np.asarray(y), 16)
            preview = ", ".join(
                f"({a:.4g}, {b:.4g})"
                for a, b in zip(xs[:16], ys[:16])
            )
            lines.append(
                f"- {ch.display_label()} [{ch.axis} axis, {ch.color}]: {preview}"
            )
        return "\n".join(lines)

    def _active_transform_context(self) -> str:
        lines = []
        if self.session is not None:
            lines.append(f"- source hash: {self.session.short_hash}")
        if self.chk_filter.isChecked():
            lines.append(
                f"- low-pass filter: {self.spn_filter_hz.value():.4g} Hz "
                f"on {self.cmb_filter_target.currentText()}")
        else:
            lines.append("- low-pass filter: off")
        for ch in self.channels:
            if not ch.enabled:
                continue
            lines.append(
                f"- {ch.display_label()}: formula={ch.formula!r}, "
                f"gain={ch.gain:g}, offset={ch.offset:g}, axis={ch.axis}")
        if getattr(self, "_sat_overlay", None):
            lines.append("- saturation overlay: visible/model estimate")
        if getattr(self, "_recon_overlay", None):
            lines.append("- RLC reconstruction overlay: visible/model estimate")
        return "\n".join(lines)

    def _paper_context(self, question: str) -> tuple[str, list[str]]:
        if self._paper_index is None:
            return "", []
        from paper_index import search
        results = search(self._paper_index, question, top_k=5)
        if not results:
            return "", []
        blocks = []
        sources = []
        seen_sources = set()
        for i, item in enumerate(results, start=1):
            base = os.path.basename(item.source)
            if base not in seen_sources:
                sources.append(base)
                seen_sources.add(base)
            blocks.append(
                f"[{i}] {item.title}\n{item.excerpt}"
            )
        context = "\n\n".join(blocks)
        if len(context) > 2400:
            context = context[:2350] + "\n...[paper context compressed]"
        return context, sources

    def _set_ai_busy(self, busy: bool):
        self.btn_ai_summary.setEnabled(not busy)
        self.btn_ai_anomaly.setEnabled(not busy)
        self.btn_ai_send.setEnabled(not busy)
        self.btn_ai_clear.setEnabled(not busy)
        self.btn_browse_gguf.setEnabled(not busy and
                                        self.cmb_backend.currentIndex() in (0, 2))
        self.btn_browse_papers.setEnabled(not busy)
        self.btn_index_papers.setEnabled(not busy)

    def _ai_trace_line(self, prompt: str, backend: str, model: str,
                       max_tokens: int) -> str:
        from version import __version__
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        system_hash = hashlib.sha256(
            CHAT_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
        source = getattr(getattr(self, "session", None), "short_hash", "n/a")
        shown_model = model or "(backend default)"
        return (
            f"AI trace: app={__version__}; backend={backend}; "
            f"model={shown_model}; prompt_sha256={prompt_hash}; "
            f"system_sha256={system_hash}; max_tokens={max_tokens}; "
            f"source={source}"
        )

    def _start_ai_request(self, question: str):
        visible = self._build_visible_context()
        if not visible:
            self._append_chat("System", "Load a file and enable channels first.")
            return
        backend = {0: "mlx", 1: "ollama", 2: "llama.cpp"}.get(
            self.cmb_backend.currentIndex(), "mlx")
        paper_text, self._pending_sources = self._paper_context(question)
        history = self._chat_history[-6:]
        prompt = []
        try:                       # accumulated lab knowledge, budgeted
            import lab_memory
            mem = lab_memory.context_block()
            if mem:
                prompt.extend([mem, ""])
        except Exception:
            pass
        prompt += [
            "Visible waveform context:",
            visible,
        ]
        if paper_text:
            prompt.extend([
                "",
                "Retrieved paper excerpts:",
                paper_text,
            ])
        if history:
            prompt.extend(["", "Recent conversation:"])
            for item in history:
                prompt.append(f"{item['role']}: {item['content']}")
        prompt.extend(["", f"User question: {question}"])

        self._chat_history.append({"role": "user", "content": question})
        self._append_chat("You", question)
        self._set_ai_busy(True)

        # two-tier dispatch: a tiny router model decides in ~0.2 s whether
        # this is a tool request (then NumPy runs it - no heavy model call)
        # or an interpretation question (then the big model answers).
        self._pending_prompt = "\n".join(prompt)
        self._pending_backend = backend
        router_model = ""
        if backend == "ollama":
            from ai_assistant import DEFAULT_ROUTER_MODEL
            router_model = DEFAULT_ROUTER_MODEL
        elif backend == "mlx":
            from ai_assistant import default_mlx_router
            router_model = default_mlx_router()
        if router_model:
            self.statusBar().showMessage("Routing…", 6000)
            self._router_worker = RouterWorker(question, router_model,
                                               backend)
            self._router_worker.done.connect(self._router_done)
            self._router_worker.start()
        else:
            self._launch_interpreter()

    def _router_done(self, raw: str):
        import json as _json
        import re as _re
        act = None
        # MLX routers have no schema enforcement and may wrap the JSON
        # in code fences or prose - extract the first {...} block
        m = _re.search(r"\{.*?\}", raw or "", _re.S)
        raw = m.group(0) if m else raw
        try:
            obj = _json.loads(raw)
            if isinstance(obj, dict) and obj.get("run") in (
                    "detect_anomalies", "channel_stats",
                    "compute_stats", "estimate_saturation",
                    "reconstruct_rlc", "zero_baseline"):
                act = obj
        except (ValueError, TypeError):
            pass            # router unavailable or no tool intent
        if act is None:
            self._launch_interpreter()
            return
        from chat_actions import run_tool
        msg = run_tool(self, act)
        self._append_chat("Tool", msg)
        self._chat_history.append({"role": "tool", "content": msg})
        self._set_ai_busy(False)
        self.statusBar().showMessage(
            "Tool executed (router) - ask a follow-up to interpret it.", 8000)

    def _launch_interpreter(self):
        self.statusBar().showMessage(
            "Thinking locally… (the first call can take a few seconds)",
            12000
        )
        max_tokens = 768
        self._pending_ai_trace = self._ai_trace_line(
            self._pending_prompt,
            self._pending_backend,
            self.ed_model.text().strip(),
            max_tokens,
        )
        self._model_worker = ModelWorker(
            self._pending_prompt,
            self.ed_model.text().strip(),
            self._pending_backend,
            CHAT_SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
        self._model_worker.done.connect(self._ai_done)
        self._model_worker.start()

    def run_ai(self):
        self._start_ai_request(
            "Summarize the visible data, compare enabled channels, and note "
            "anything unusual or worth checking."
        )

    def run_anomaly_scan(self):
        """Deterministic anomaly scan of the visible window - no LLM call.
        The result is posted to the chat AND the history, so a follow-up
        question like 'what could cause these?' lets the model interpret
        the exact numbers instead of recomputing anything."""
        if self.data is None:
            self._append_chat("System", "Load a file and enable channels first.")
            return
        from chat_actions import run_tool
        msg = run_tool(self, {"run": "detect_anomalies"})
        self._append_chat("Tool", msg)
        self._chat_history.append({"role": "tool", "content": msg})
        self.statusBar().showMessage(
            "Anomaly scan done - ask the chat to explain likely causes.",
            8000)

    @staticmethod
    def _trigger_t0(x: np.ndarray, y: np.ndarray) -> float | None:
        """Trigger time: first crossing of 50% of the peak deviation from
        the early baseline. Used to align overlay shots at t=0."""
        base = float(np.nanmedian(y[: max(16, len(y) // 20)]))
        dev = np.abs(y - base)
        pk = float(np.nanmax(dev))
        if pk <= 0:
            return None
        idx = np.flatnonzero(dev >= 0.5 * pk)
        return float(x[idx[0]]) if idx.size else None

    def add_overlay_shots(self):
        """Overlay other shots on the current plot for comparison
        (e.g. the charging-voltage step series). Each file's channels are
        matched BY NAME to the currently enabled channels and pass
        through the same formula/gain/offset conversion; traces are
        trigger-aligned to the base shot and drawn semi-transparent with
        'file:channel' legend names."""
        if self.data is None:
            QMessageBox.information(self, "Overlay",
                                    "Load a base CSV first.")
            return
        enabled = [ch for ch in self.channels if ch.enabled]
        if not enabled:
            QMessageBox.information(self, "Overlay",
                                    "Enable at least one channel first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add overlay shots", "",
            "Data files (*.csv *.CSV *.txt *.tsv);;All files (*)")
        if not paths:
            return
        import os
        xfac = self.spn_xscale.value()
        try:
            t0_main = self._trigger_t0(self._x(),
                                       self._channel_data(enabled[0]))
        except Exception:
            t0_main = None
        if not hasattr(self, "_shot_overlays"):
            self._shot_overlays = []
        n = 0
        for p in paths:
            try:
                d = load_csv(p)
            except Exception as e:
                self._append_chat(
                    "System", f"Overlay skipped {os.path.basename(p)}: {e}")
                continue
            tag = os.path.splitext(os.path.basename(p))[0]
            raw_x = d.df.iloc[:, 0].to_numpy(np.float64)
            x_disp = raw_x * xfac
            shift = 0.0
            if t0_main is not None and enabled[0].name in d.df.columns:
                y_ref = enabled[0].convert(
                    d.df[enabled[0].name].to_numpy(np.float64),
                    raw_x, raw_x * 1000.0)
                t0_ov = self._trigger_t0(x_disp, y_ref)
                if t0_ov is not None:
                    shift = t0_main - t0_ov
            for ch in enabled:
                if ch.name not in d.df.columns:
                    continue
                y = ch.convert(d.df[ch.name].to_numpy(np.float64),
                               raw_x, raw_x * 1000.0)
                xs, ys = minmax_decimate(x_disp + shift, y, 8000)
                col = QColor(ch.color)
                col.setAlpha(140)
                item = pg.PlotDataItem(xs, ys, pen=pg.mkPen(col, width=1),
                                       name=f"{tag}:{ch.name}")
                self.pi.addItem(item)
                self._shot_overlays.append(item)
            n += 1
        self.statusBar().showMessage(
            f"{n} overlay shot(s) added, trigger-aligned to the base "
            f"shot. File > Clear overlay shots removes them.", 9000)

    def clear_overlay_shots(self):
        for it in getattr(self, "_shot_overlays", []):
            try:
                self.pi.removeItem(it)
            except Exception:
                pass
        self._shot_overlays = []

    def apply_sat_overlay(self):
        """Draw (or clear) the saturation-fit overlay: dashed rise/droop
        projection lines and a marker at the reconstructed peak corner.
        Data comes from the last estimate_saturation run; the checkbox
        toggles visibility without re-running the fit."""
        for it in getattr(self, "_sat_overlay_items", []):
            try:
                self.pi.removeItem(it)
            except Exception:
                pass
        self._sat_overlay_items = []
        ov = getattr(self, "_sat_overlay", None)
        if not ov or not self.chk_sat_overlay.isChecked():
            return
        pen = pg.mkPen("#d55e00", width=2, style=Qt.DashLine)  # Wong orange
        named = False
        for key in ("rise", "droop"):
            f = ov.get(key)
            if not f:
                continue
            xs = [f["x0"], f["x1"]]
            ys = [f["a"] * xs[0] + f["b"], f["a"] * xs[1] + f["b"]]
            item = pg.PlotDataItem(
                xs, ys, pen=pen,
                name=None if named else "Saturation fit")
            named = True
            self.pi.addItem(item)
            self._sat_overlay_items.append(item)
        inter = ov.get("intersection")
        if inter:
            tx, ix = inter
            dot = pg.ScatterPlotItem([tx], [ix], size=11, symbol="d",
                                     brush=pg.mkBrush("#d55e00"),
                                     pen=pg.mkPen("k"))
            txt = pg.TextItem(f"est. peak {ix:,.0f}", color="#d55e00",
                              anchor=(0.5, 1.3))
            txt.setPos(tx, ix)
            self.pi.addItem(dot)
            self.pi.addItem(txt)
            self._sat_overlay_items += [dot, txt]

    def apply_recon_overlay(self):
        """Draw (or clear) the RLC reconstruction: solid curve + dotted
        95% band, over the measured traces for visual comparison."""
        for it in getattr(self, "_recon_overlay_items", []):
            try:
                self.pi.removeItem(it)
            except Exception:
                pass
        self._recon_overlay_items = []
        cv = getattr(self, "_recon_overlay", None)
        if not cv or not self.chk_recon_overlay.isChecked():
            return
        col = "#009e73"                       # Wong bluish green
        mid = pg.PlotDataItem(cv["t"], cv["y"],
                              pen=pg.mkPen(col, width=2),
                              name="RLC reconstruction")
        lo = pg.PlotDataItem(cv["t"], cv["lo"],
                             pen=pg.mkPen(col, width=1, style=Qt.DotLine))
        hi = pg.PlotDataItem(cv["t"], cv["hi"],
                             pen=pg.mkPen(col, width=1, style=Qt.DotLine))
        band = pg.FillBetweenItem(lo, hi, brush=pg.mkBrush(0, 158, 115, 40))
        for it in (band, lo, hi, mid):
            self.pi.addItem(it)
        self._recon_overlay_items = [band, lo, hi, mid]

    def _infer_hidden_peak_channel(self) -> Channel | None:
        """Pick the likely saturated current monitor from enabled channels.

        Priority is intentionally conservative: calibrated BBCM/busbar
        channels first, then the selected enabled current-like row, then the
        first enabled current-like trace. The user can still override by
        selecting/changing channels manually before pressing the button.
        """
        enabled = [ch for ch in self.channels if ch.enabled]
        if not enabled:
            return None

        def score(ch: Channel) -> int:
            text = " ".join([
                ch.name, ch.display_label(), ch.preset_name,
                ch.formula, ch.unit,
            ]).lower()
            value = 0
            if "bbcm" in text:
                value += 120
            if "busbar" in text or "bus bar" in text:
                value += 100
            if "current" in text or ch.unit.lower() == "a" or "(a)" in text:
                value += 30
            if "pearson" in text:
                value -= 80
            if "peak detect" in text:
                value -= 120
            return value

        ranked = sorted(enabled, key=score, reverse=True)
        if score(ranked[0]) > 0:
            return ranked[0]

        row = self._selected_channel_row()
        if 0 <= row < len(self.channels) and self.channels[row].enabled:
            return self.channels[row]
        return enabled[0]

    def _infer_hidden_peak_reference(self, target: Channel) -> Channel | None:
        """Choose a likely clean current reference, usually Pearson CH2."""
        candidates = [ch for ch in self.channels
                      if ch.enabled and ch is not target]
        if not candidates:
            return None

        def score(ch: Channel) -> int:
            text = " ".join([
                ch.name, ch.display_label(), ch.preset_name,
                ch.formula, ch.unit,
            ]).lower()
            value = 0
            if "pearson" in text:
                value += 100
            if ch.unit.lower() == "a" or "(a)" in text or "current" in text:
                value += 40
            if "bbcm" in text or "busbar" in text or "bus bar" in text:
                value -= 60
            if "peak detect" in text:
                value -= 100
            return value

        ref = max(candidates, key=score)
        return ref if score(ref) > 0 else None

    def _infer_hidden_peak_sat_level(self, target: Channel) -> tuple[float, str]:
        """Infer a censoring level in display units from calibration state."""
        text = " ".join([
            target.name, target.display_label(), target.preset_name,
            target.formula, target.unit,
        ]).lower()
        if "bbcm" in text or "busbar" in text or "bus bar" in text:
            # Jean's BBCM presets use gain as the module-count/summing factor.
            # A single module is ~1500 A; gain=4 implies the 6 kA benchmark.
            modules = max(1.0, abs(float(target.gain)))
            sat = 1500.0 * modules
            return sat, (
                f"BBCM/busbar preset with gain {target.gain:g} "
                f"-> {sat:g} A lower-bound censoring level")
        return float(USER_DEFAULTS["sat_level"]), (
            f"rig default {USER_DEFAULTS['sat_level']:g} A censoring level")

    def _infer_hidden_peak_window(self, target: Channel) -> tuple[float, float, str]:
        """Use the visible range, trimming a detected switch-off fall if seen."""
        x = self._x()
        if x is None:
            raise RuntimeError("Load a CSV first.")
        (x0, x1) = self.pi.vb.viewRange()[0]
        m = (x >= x0) & (x <= x1)
        xv = np.asarray(x[m], dtype=np.float64)
        yv = np.asarray(self._channel_data(target)[m], dtype=np.float64)
        if xv.size < 256:
            return float(x0), float(x1), "visible window"

        finite = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[finite], yv[finite]
        if xv.size < 256:
            return float(x0), float(x1), "visible window"

        sgn = 1.0 if abs(np.nanmax(yv)) >= abs(np.nanmin(yv)) else -1.0
        ys = sgn * yv
        pk = float(np.nanmax(ys))
        if not np.isfinite(pk) or pk <= 0:
            return float(x0), float(x1), "visible window"

        # Work on a uniformly sub-sampled trace so huge files stay instant.
        step = max(1, xv.size // 5000)
        xd, yd = xv[::step], ys[::step]
        if yd.size < 32:
            return float(x0), float(x1), "visible window"
        w = min(21, max(5, (yd.size // 250) | 1))
        if w > 3:
            kernel = np.ones(w, dtype=np.float64) / w
            yd_s = np.convolve(yd, kernel, mode="same")
        else:
            yd_s = yd
        peak_idx = int(np.nanargmax(yd_s))
        dy = np.diff(yd_s)
        # Switch-off is a cliff compared with the natural droop. Requiring a
        # subsequent fall below half-peak prevents normal droop from being
        # mistaken for the end of the fit window.
        drop_threshold = -0.004 * pk
        max_ahead = max(8, yd_s.size // 40)
        for i in range(peak_idx + 1, len(dy)):
            if dy[i] > drop_threshold:
                continue
            ahead = yd_s[i + 1:min(len(yd_s), i + 1 + max_ahead)]
            if ahead.size and float(np.nanmin(ahead)) < 0.5 * pk:
                return float(xv[0]), float(xd[i]), (
                    f"visible window trimmed before switch-off at {xd[i]:.4g}")
        return float(xv[0]), float(xv[-1]), "visible window"

    @staticmethod
    def _format_time_windows(windows: list[tuple[float, float]] | None) -> str:
        if not windows:
            return ""
        return ", ".join(f"{a:g}:{b:g}" for a, b in windows)

    @staticmethod
    def _parse_time_windows(text: str) -> list[tuple[float, float]]:
        """Parse user windows like '0:5, 40:150' in display X units."""
        text = (text or "").strip()
        if not text:
            return []

        number = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        windows: list[tuple[float, float]] = []
        for raw_part in re.split(r"[;,]", text):
            part = raw_part.strip()
            if not part:
                continue
            match = re.match(
                rf"^\s*({number})\s*(?::|\.{{2}}|\bto\b)\s*({number})\s*$",
                part,
                flags=re.IGNORECASE,
            )
            if match is None:
                # Convenience for positive-only "0-5" style entries.
                match = re.match(
                    rf"^\s*([+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*-\s*"
                    rf"([+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$",
                    part,
                )
            if match is None:
                raise ValueError(
                    f"Could not parse trusted region '{part}'. Use "
                    "examples like 0:5, 40:150.")
            lo, hi = float(match.group(1)), float(match.group(2))
            if lo == hi:
                raise ValueError(f"Trusted region '{part}' has zero width.")
            windows.append(tuple(sorted((lo, hi))))
        return windows

    def _hidden_peak_settings_dialog(
        self,
        target: Channel,
        ref: Channel | None,
        sat_level: float,
        t0: float,
        t1: float,
    ) -> dict | None:
        """Beginner-friendly confirmation dialog before reconstruction."""
        enabled = [ch for ch in self.channels if ch.enabled]
        dlg = QDialog(self)
        dlg.setWindowTitle("Recover hidden peak")
        outer = QVBoxLayout(dlg)

        intro = QLabel(
            "Confirm the reconstruction assumptions. The source CSV remains "
            "read-only; this only creates reversible in-memory overlays."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        form = QFormLayout()
        cmb_target = QComboBox()
        for ch in enabled:
            cmb_target.addItem(ch.display_label(), ch.name)
        idx = cmb_target.findData(target.name)
        if idx >= 0:
            cmb_target.setCurrentIndex(idx)

        ed_sat = QLineEdit(f"{sat_level:g}")
        ed_fit_start = QLineEdit(f"{t0:g}")
        ed_fit_end = QLineEdit(f"{t1:g}")

        cmb_ref = QComboBox()
        cmb_ref.addItem("(none)", "")
        for ch in enabled:
            if ch is not target:
                cmb_ref.addItem(ch.display_label(), ch.name)
        if ref is not None:
            idx = cmb_ref.findData(ref.name)
            if idx >= 0:
                cmb_ref.setCurrentIndex(idx)

        ed_ref_start = QLineEdit(f"{t0:g}")
        ed_ref_end = QLineEdit(f"{USER_DEFAULTS['ref_end_ms']:g}")
        ed_trusted = QLineEdit()
        ed_trusted.setPlaceholderText(
            "Optional, e.g. 0:5, 40:150. Blank = all non-censored samples."
        )

        form.addRow("Measured channel:", cmb_target)
        form.addRow("Saturation/lower-bound level:", ed_sat)
        form.addRow("Fit start:", ed_fit_start)
        form.addRow("Fit end:", ed_fit_end)
        form.addRow("Reference channel:", cmb_ref)
        form.addRow("Reference valid from:", ed_ref_start)
        form.addRow("Reference valid to:", ed_ref_end)
        form.addRow("Trusted target regions:", ed_trusted)
        outer.addLayout(form)

        help_text = QLabel(
            "Use trusted target regions when a sensor is accurate only in "
            "specific windows. Example: if the Pearson is reliable until "
            "5 ms and the BBCM becomes reliable again after 40 ms, enter "
            "`0:5, 40:150`. Saturated samples above the limit still act as "
            "lower bounds, not exact data."
        )
        help_text.setWordWrap(True)
        outer.addWidget(help_text)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Run recovery")
        outer.addWidget(buttons)

        result: dict = {}

        def accept_if_valid():
            try:
                target_ch = self._channel_by_name(cmb_target.currentData())
                if target_ch is None:
                    raise ValueError("Choose a measured channel.")
                ref_ch = self._channel_by_name(cmb_ref.currentData()) \
                    if cmb_ref.currentData() else None
                if ref_ch is target_ch:
                    raise ValueError(
                        "Reference channel must be different from the "
                        "measured channel.")
                sat = float(ed_sat.text().strip())
                fit_start = float(ed_fit_start.text().strip())
                fit_end = float(ed_fit_end.text().strip())
                if fit_start == fit_end:
                    raise ValueError("Fit start and end cannot be equal.")
                trusted = self._parse_time_windows(ed_trusted.text())
                ref_start = float(ed_ref_start.text().strip()) \
                    if ref_ch is not None else None
                ref_end = float(ed_ref_end.text().strip()) \
                    if ref_ch is not None else None
                if ref_ch is not None and ref_start == ref_end:
                    raise ValueError(
                        "Reference valid start/end cannot be equal.")
            except ValueError as exc:
                QMessageBox.warning(dlg, "Recover hidden peak", str(exc))
                return

            result.update({
                "target": target_ch,
                "reference": ref_ch,
                "sat_level": sat,
                "fit_window": tuple(sorted((fit_start, fit_end))),
                "ref_window": (
                    tuple(sorted((ref_start, ref_end)))
                    if ref_ch is not None else None
                ),
                "trusted_windows": trusted,
            })
            dlg.accept()

        buttons.accepted.connect(accept_if_valid)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return None
        return result

    def _hidden_peak_actions(self) -> tuple[list[tuple[str, dict]], str]:
        target = self._infer_hidden_peak_channel()
        if target is None:
            return [], "Load a file and enable a current/BBCM channel first."
        sat_level, sat_reason = self._infer_hidden_peak_sat_level(target)
        t0, t1, window_reason = self._infer_hidden_peak_window(target)
        ref = self._infer_hidden_peak_reference(target)
        settings = self._hidden_peak_settings_dialog(
            target, ref, sat_level, t0, t1)
        if settings is None:
            return [], "Hidden peak recovery canceled."
        target = settings["target"]
        ref = settings["reference"]
        sat_level = settings["sat_level"]
        t0, t1 = settings["fit_window"]
        trusted_windows = settings["trusted_windows"]
        auto_sat, auto_sat_reason = self._infer_hidden_peak_sat_level(target)
        if math.isclose(sat_level, auto_sat, rel_tol=1e-9, abs_tol=1e-9):
            sat_reason = auto_sat_reason
        else:
            sat_reason = (
                f"user-entered setting; auto estimate for selected channel "
                f"would be {auto_sat:g} A")

        common = {
            "channel": target.display_label(),
            "sat_level": sat_level,
        }
        recon = {
            "run": "reconstruct_rlc",
            **common,
            "t_start": t0,
            "t_end": t1,
        }
        ref_note = "no reference channel inferred"
        if ref is not None:
            recon["ref_channel"] = ref.display_label()
            ref_start, ref_end = settings["ref_window"]
            recon["ref_start"] = ref_start
            recon["ref_end"] = ref_end
            ref_note = (
                f"{ref.display_label()} trusted from "
                f"{ref_start:g} to {ref_end:g} display units")
        if trusted_windows:
            recon["trusted_windows"] = trusted_windows
            trusted_note = self._format_time_windows(trusted_windows)
        else:
            trusted_note = "auto: all non-censored target samples"

        summary = (
            "Recover hidden peak settings:\n"
            f"  target: {target.display_label()} "
            f"(preset {target.preset_name}, gain {target.gain:g}, "
            f"offset {target.offset:g})\n"
            f"  censoring: {sat_level:g} A lower-bound level "
            f"({sat_reason})\n"
            f"  fit window: {t0:.5g} to {t1:.5g} ({window_reason})\n"
            f"  trusted target regions: {trusted_note}\n"
            f"  reference: {ref_note}\n"
            "  policy: original CSV untouched; overlays/transforms are "
            "in-memory display estimates."
        )
        return [
            ("Saturation estimate", {"run": "estimate_saturation", **common}),
            ("Hidden-peak reconstruction", recon),
        ], summary

    def run_hidden_peak_recovery(self):
        """One-click censored recovery for saturated BBCM/busbar pulses."""
        if self.data is None:
            self._append_chat("System",
                              "Load a file and enable channels first.")
            return
        from chat_actions import run_tool
        actions, summary = self._hidden_peak_actions()
        if not actions:
            self._append_chat("System", summary)
            return
        self._append_chat("System", summary)
        for name, act in actions:
            self.statusBar().showMessage(f"{name}…", 4000)
            try:
                msg = run_tool(self, act)
            except Exception as e:
                msg = f"{name} failed: {e!r}"
            self._append_chat("Tool", f"[{name}]\n{msg}")
            self._chat_history.append(
                {"role": "tool", "content": f"[{name}] {msg}"})
        self.statusBar().showMessage(
            "Hidden peak recovery done - overlays drawn; original CSV "
            "untouched.", 9000)

    def run_rlc_reconstruct(self):
        """Backward-compatible wrapper for older shortcuts/tests."""
        self.run_hidden_peak_recovery()

    def export_obsidian_note(self):
        """Write a connected session note ([[wikilinks]]) into the user's
        Obsidian vault: tools used + deterministic 'why', verbatim tool
        outputs from this session, source-file hash. The note is a
        faithful record built from history - no model involvement."""
        if self.data is None:
            QMessageBox.information(self, "Obsidian note",
                                    "Load a file first.")
            return
        import obsidian_notes as obs
        vault = obs.get_vault()
        if not vault or not os.path.isdir(vault):
            vault = QFileDialog.getExistingDirectory(
                self, "Choose your Obsidian vault folder",
                os.path.expanduser("~"))
            if not vault:
                return
            obs.set_vault(vault)
        sess = getattr(self, "session", None)
        shot = os.path.splitext(os.path.basename(self.data.path))[0]
        tool_events = [h["content"] for h in self._chat_history
                       if h.get("role") == "tool"]
        comment, _ok = "", True
        try:
            from PySide6.QtWidgets import QInputDialog
            comment, _ok = QInputDialog.getMultiLineText(
                self, "Your interpretation (optional)",
                "Notes to include under 'My interpretation':")
        except Exception:
            pass
        md = obs.session_note_markdown(
            shot_name=shot, source_path=self.data.path,
            source_hash=getattr(sess, "source_hash", "n/a"),
            channels=[ch.name for ch in self.channels if ch.enabled],
            tool_events=tool_events,
            ai_events=getattr(self, "_ai_trace_events", []),
            user_comment=comment or "")
        path = obs.write_note(vault, f"Shot {shot}", md)
        self.statusBar().showMessage(
            f"Obsidian note written: {os.path.basename(path)}", 8000)

    def run_shot_pipeline(self):
        """Jean's one-click analysis: the safe default sequence, all
        deterministic, results posted to chat + history so the model can
        interpret on the next question."""
        if self.data is None:
            self._append_chat("System",
                              "Load a file and enable channels first.")
            return
        from chat_actions import run_tool
        x0, x1 = self.pi.vb.viewRange()[0]
        steps = [
            ("Zero baselines", {"run": "zero_baseline"}),
            ("Statistics", {"run": "compute_stats"}),
            ("Anomaly scan", {"run": "detect_anomalies"}),
            ("Saturation estimate",
             {"run": "estimate_saturation",
              "sat_level": USER_DEFAULTS["sat_level"]}),
            ("RLC reconstruction",
             {"run": "reconstruct_rlc",
              "sat_level": USER_DEFAULTS["sat_level"],
              "t_start": x0, "t_end": x1,
              "ref_end": USER_DEFAULTS["ref_end_ms"]}),
        ]
        self._append_chat("System",
                          "Running shot pipeline (5 deterministic steps)…")
        for name, act in steps:
            self.statusBar().showMessage(f"Pipeline: {name}…", 4000)
            try:
                msg = run_tool(self, act)
            except Exception as e:
                msg = f"{name} failed: {e!r}"
            self._append_chat("Tool", f"[{name}]\n{msg}")
            self._chat_history.append(
                {"role": "tool", "content": f"[{name}] {msg}"})
        self.statusBar().showMessage(
            "Pipeline done - overlays drawn; ask the chat to interpret.",
            10000)

    def run_saturation_estimate(self):
        """Deterministic saturation-recovery scan of the visible window
        (saturation_recovery.py, Change Set 29) - no LLM call. Output goes
        to chat + history so a follow-up question can interpret it."""
        if self.data is None:
            self._append_chat("System",
                              "Load a file and enable channels first.")
            return
        from chat_actions import run_tool
        msg = run_tool(self, {"run": "estimate_saturation"})
        self._append_chat("Tool", msg)
        self._chat_history.append({"role": "tool", "content": msg})
        self.statusBar().showMessage(
            "Saturation estimate done - ask the chat to interpret it.",
            8000)

    def _send_ai_prompt(self):
        question = self.ed_ai_prompt.toPlainText().strip()
        if not question:
            return
        self.ed_ai_prompt.clear()
        # lab-memory chat commands (deterministic, no model needed):
        #   remember: <fact>     -> append to the journal
        #   compress memory      -> LLM re-compresses the digest
        low = question.lower()
        if low.startswith("remember:") or low.startswith("remember "):
            import lab_memory
            fact = question.split(":", 1)[-1].strip() if ":" in question \
                else question[len("remember"):].strip()
            n = lab_memory.add_entry(fact, source="chat")
            self._append_chat("System",
                              f"Remembered ({n} entries in the journal). "
                              "Say 'compress memory' to fold it into the "
                              "digest the assistant carries.")
            return
        if low in ("compress memory", "rebuild memory"):
            import lab_memory
            from ai_assistant import ask_model
            backend = {0: "mlx", 1: "ollama", 2: "llama.cpp"}.get(
                self.cmb_backend.currentIndex(), "ollama")
            self._append_chat("System", "Compressing lab memory…")
            try:
                d = lab_memory.rebuild_digest(
                    lambda p: ask_model(p, model=self.ed_model.text()
                                        .strip(), backend=backend,
                                        max_tokens=1200))
                self._append_chat("System",
                                  f"Digest rebuilt ({len(d)} chars). It "
                                  "now rides along in every question.")
            except Exception as e:
                self._append_chat("System", f"Compress failed: {e}")
            return
        self._start_ai_request(question)

    def _ai_done(self, text: str):
        from chat_actions import process_reply
        clean, applied, tool_msgs = process_reply(self, text)
        lines = [clean] if clean else []
        if self._pending_sources:
            lines.append("")
            lines.append("Paper context used: " + ", ".join(self._pending_sources))
        answer = "\n".join(lines).strip() or "(actions only)"
        self._chat_history.append({"role": "assistant", "content": answer})
        self._append_chat("Assistant", answer)
        trace = getattr(self, "_pending_ai_trace", "")
        if trace:
            self._ai_trace_events.append(trace)
            self._append_chat("System", trace)
            self._pending_ai_trace = ""
        if applied:
            note = "Applied: " + "; ".join(applied)
            self._append_chat("System", note)
            self._chat_history.append({"role": "tool", "content": note})
        for msg in tool_msgs:
            self._append_chat("Tool", msg)
            # store tool output in history so the model can interpret it
            # on the user's next turn ("explain those anomalies")
            self._chat_history.append({"role": "tool", "content": msg})
        self._pending_sources = []
        self._set_ai_busy(False)
        self.statusBar().clearMessage()

    # ---------------------------------------------------------- export -----
    def export_pub(self):
        if self.data is None:
            QMessageBox.information(self, "Export", "Load a CSV first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", "figure.svg",
            "SVG vector (*.svg);;JPG image (*.jpg);;PNG image (*.png);;"
            "PDF vector (*.pdf)")
        if not path:
            return
        x = self._x()
        traces = []
        for ch in self.channels:
            if not ch.enabled:
                continue
            try:
                y = self._channel_data(ch)
            except Exception as exc:
                QMessageBox.critical(
                    self, "Export failed",
                    f"Cannot export {ch.display_label()}:\n{exc}"
                )
                return
            traces.append(ExportTrace(
                x=x, y=y,
                label=ch.display_label(), color=ch.color, axis=ch.axis))
        if not traces:
            QMessageBox.information(self, "Export", "No channels enabled.")
            return
        width = [89.0, 120.0, 183.0][self.cmb_width.currentIndex()]
        (x0, x1), (y0, y1) = self.pi.vb.viewRange()
        r0r1 = self.vb_right.viewRange()[1] if self.right_axis.isVisible() \
            else None
        opts = ExportOptions(
            width_mm=width, height_mm=self.spn_height.value(),
            dpi=self.spn_dpi.value(), title=self.ed_title.text(),
            xlabel=self.ed_xlabel.text(), ylabel_left=self.ed_yllabel.text(),
            ylabel_right=self.ed_yrlabel.text(),
            xlim=(x0, x1), ylim_left=(y0, y1), ylim_right=r0r1,
            align_zero=self.chk_zero.isChecked(),
            show_grid=self.chk_grid.isChecked(),
            top_axis=self.chk_top.isChecked(),
            top_scale=self.spn_topscale.value(),
            top_label=self.ed_toplabel.text(),
            legend_loc=self.cmb_legend.currentText(),
            max_points=0 if self.chk_fullres.isChecked() else 200_000)
        try:
            export_figure(traces, opts, path)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(self, "Export", f"Saved:\n{path}")


def main():
    app = QApplication(sys.argv)
    # minimalist rounded theme (palette-aware: works in dark and light
    # mode); missing/broken stylesheet must never block startup
    try:
        qss = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "style.qss")
        with open(qss, "r") as fh:
            app.setStyleSheet(fh.read())
    except Exception:
        pass
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
