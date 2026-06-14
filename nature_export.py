"""
nature_export.py — Publication-ready figure export (Nature-journal style).

Renders the current view with matplotlib using Nature figure conventions:
Arial/Helvetica, 7–8 pt type, thin spines, single-column (89 mm) or
double-column (183 mm) widths, vector SVG with editable text, or high-DPI JPG.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from csv_loader import minmax_decimate

MM = 1 / 25.4  # mm → inch

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "axes.linewidth": 0.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "lines.linewidth": 0.75,
    "axes.unicode_minus": True,
    "svg.fonttype": "none",   # keep text editable in Illustrator/Inkscape
    "pdf.fonttype": 42,
}


@dataclass
class ExportTrace:
    x: np.ndarray
    y: np.ndarray
    label: str
    color: str
    axis: str = "left"          # "left" | "right"
    linestyle: str = "-"


@dataclass
class ExportOptions:
    width_mm: float = 89.0       # 89 = single col, 183 = double col (Nature)
    height_mm: float = 60.0
    dpi: int = 600
    title: str = ""
    xlabel: str = "Time (ms)"
    ylabel_left: str = ""
    ylabel_right: str = ""
    xlim: tuple | None = None
    ylim_left: tuple | None = None
    ylim_right: tuple | None = None
    align_zero: bool = True
    show_grid: bool = False
    top_axis: bool = False
    top_scale: float = 1.0       # top tick value = bottom value * scale + offset
    top_offset: float = 0.0
    top_label: str = ""
    legend_loc: str = "best"
    max_points: int = 200_000    # per-trace decimation cap (0 = full resolution)
    right_color: str = "#D55E00" # tint right axis like the slides' orange


def _align_zero_limits(lims_left, lims_right):
    """Recompute right-axis limits so y=0 sits at the same height on both."""
    l0, l1 = lims_left
    r0, r1 = lims_right
    if l1 <= l0 or r1 <= r0:
        return lims_right
    frac = (0.0 - l0) / (l1 - l0)
    frac = min(max(frac, 1e-3), 1 - 1e-3)
    span = 0.0
    if r1 > 0:
        span = max(span, r1 / (1 - frac))
    if r0 < 0:
        span = max(span, -r0 / frac)
    if span <= 0:
        span = (r1 - r0) or 1.0
    return (-frac * span, (1 - frac) * span)


def export_figure(traces: list[ExportTrace], opts: ExportOptions, path: str):
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(
            figsize=(opts.width_mm * MM, opts.height_mm * MM), dpi=opts.dpi
        )
        ax_r = None
        if any(t.axis == "right" for t in traces):
            ax_r = ax.twinx()

        handles, labels = [], []
        for t in traces:
            x, y = t.x, t.y
            if opts.max_points and len(x) > opts.max_points:
                x, y = minmax_decimate(np.asarray(x), np.asarray(y),
                                       opts.max_points)
            target = ax_r if (t.axis == "right" and ax_r is not None) else ax
            (ln,) = target.plot(x, y, color=t.color, label=t.label,
                                linestyle=t.linestyle)
            handles.append(ln)
            labels.append(t.label)

        if opts.xlim:
            ax.set_xlim(*opts.xlim)
        if opts.ylim_left:
            ax.set_ylim(*opts.ylim_left)
        if ax_r is not None:
            if opts.ylim_right:
                ax_r.set_ylim(*opts.ylim_right)
            if opts.align_zero:
                ax_r.set_ylim(*_align_zero_limits(ax.get_ylim(),
                                                  ax_r.get_ylim()))
        ax.minorticks_on()
        if ax_r is not None:
            ax_r.minorticks_on()

        ax.set_xlabel(opts.xlabel)
        ax.set_ylabel(opts.ylabel_left)
        if opts.title:
            ax.set_title(opts.title, fontweight="bold")
        if opts.show_grid:
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4,
                    color="0.5")

        # spine policy: clean Nature look — hide unused sides
        ax.spines["top"].set_visible(opts.top_axis)
        if ax_r is None:
            ax.spines["right"].set_visible(False)
        else:
            ax_r.set_ylabel(opts.ylabel_right, color=opts.right_color)
            ax_r.tick_params(axis="y", colors=opts.right_color)
            ax_r.spines["right"].set_color(opts.right_color)
            ax_r.spines["top"].set_visible(opts.top_axis)
            ax_r.spines["left"].set_visible(False)

        if opts.top_axis:
            sec = ax.secondary_xaxis(
                "top",
                functions=(lambda v: v * opts.top_scale + opts.top_offset,
                           lambda v: (v - opts.top_offset) / opts.top_scale
                           if opts.top_scale else v),
            )
            if opts.top_label:
                sec.set_xlabel(opts.top_label)

        if labels:
            kw = dict(borderpad=0.2, labelspacing=0.25, handlelength=1.6)
            if opts.legend_loc == "outside right":
                kw.update(loc="upper left", bbox_to_anchor=(1.12, 1.0))
            else:
                kw.update(loc=opts.legend_loc)
            ax.legend(handles, labels, **kw)

        fig.tight_layout(pad=0.4)
        save_kw = {"dpi": opts.dpi}
        if path.lower().endswith((".jpg", ".jpeg")):
            save_kw.update(pil_kwargs={"quality": 95})
        fig.savefig(path, **save_kw)
        plt.close(fig)
    return path
