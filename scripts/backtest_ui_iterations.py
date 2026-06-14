#!/usr/bin/env python3
"""
Run three UI backtest iterations with a synthetic oscilloscope CSV.

The goal is not a full GUI test suite yet. It creates reproducible screenshots
and a short assessment report so UI changes can be judged visually.
"""
from __future__ import annotations

import os
import sys
import textwrap

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import MainWindow  # noqa: E402
from csv_loader import load_csv  # noqa: E402


def make_csv(path: str):
    t = np.linspace(-0.004, 0.030, 60_000)
    pulse = ((t >= 0.002) & (t <= 0.018)).astype(float)
    rise = 1.0 / (1.0 + np.exp(-(t - 0.003) / 0.00035))
    fall = 1.0 / (1.0 + np.exp((t - 0.018) / 0.00045))
    current = 1450.0 * rise * fall
    ripple = 55.0 * np.sin(2 * np.pi * 18_000 * t) * pulse
    busbar_v = 2.5 + (current + ripple) / 750.0
    pearson_a = current + 18.0 * np.sin(2 * np.pi * 40_000 * t) * pulse
    control = 5.0 * pulse
    rogowski = 0.03 * np.gradient(current, t) / np.nanmax(np.gradient(current, t))
    df = pd.DataFrame({
        "TIME": t,
        "CH1 Busbar V": busbar_v,
        "CH2 Pearson A": pearson_a,
        "CH3 Control V": control,
        "CH4 Rogowski V": rogowski,
    })
    df.to_csv(path, index=False)


def configure_channels(win: MainWindow):
    for ch in win.channels:
        ch.enabled = False
    lookup = {ch.name: ch for ch in win.channels}
    lookup["CH1 Busbar V"].enabled = True
    lookup["CH1 Busbar V"].axis = "left"
    lookup["CH1 Busbar V"].formula = "(x - 2.5) * 1500 / 2"
    lookup["CH1 Busbar V"].unit = "A"
    lookup["CH1 Busbar V"].label = "Busbar current"
    lookup["CH2 Pearson A"].enabled = True
    lookup["CH2 Pearson A"].axis = "left"
    lookup["CH2 Pearson A"].unit = "A"
    lookup["CH2 Pearson A"].label = "Pearson reference"
    lookup["CH3 Control V"].enabled = True
    lookup["CH3 Control V"].axis = "right"
    lookup["CH3 Control V"].unit = "V"
    lookup["CH3 Control V"].label = "Control signal"


def apply_iteration(win: MainWindow, idx: int) -> str:
    if idx == 1:
        win.cmb_journal_style.setCurrentText("Nature Physics")
        win.ed_title.setText("Iteration 1 - Baseline scientific plot")
        win.chk_filter.setChecked(False)
        win.chk_zero.setChecked(True)
        win.pi.vb.setXRange(-2, 25, padding=0)
        win.pi.vb.setYRange(-100, 1700, padding=0)
        return "Baseline: validates the copied app still loads, plots, and labels a shot."

    if idx == 2:
        win.cmb_journal_style.setCurrentText("Teaching / Slides")
        win.ed_title.setText("Iteration 2 - Student-friendly teaching view")
        win.chk_filter.setChecked(True)
        win.spn_filter_hz.setValue(15_000.0)
        win.cmb_filter_target.setCurrentText("Current-like channels")
        win.chk_grid.setChecked(True)
        win.pi.vb.setXRange(-1, 23, padding=0)
        win.pi.vb.setYRange(-50, 1600, padding=0)
        return (
            "Student-friendly: larger fonts, grid visible, current-like "
            "low-pass filtering enabled."
        )

    win.cmb_journal_style.setCurrentText("Nature Communications")
    win.ed_title.setText("Iteration 3 - Publication-ready current comparison")
    win.chk_filter.setChecked(True)
    win.spn_filter_hz.setValue(10_000.0)
    win.cmb_filter_target.setCurrentText("Current-like channels")
    win.chk_grid.setChecked(False)
    win.chk_zero.setChecked(True)
    win.pi.vb.setXRange(-1, 22, padding=0)
    win.pi.vb.setYRange(-50, 1600, padding=0)
    return (
        "Publication-ready: smaller journal typography, 10 kHz filter, "
        "aligned dual axes, and cleaner grid policy."
    )


def main():
    os.makedirs(os.path.join(ROOT, "backtests"), exist_ok=True)
    csv_path = os.path.join(ROOT, "backtests", "synthetic_scope_shot.csv")
    make_csv(csv_path)

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.resize(1700, 940)
    win.show()
    win.data = load_csv(csv_path)
    win.lbl_file.setText("synthetic_scope_shot.csv - UI backtest")
    win._init_channels()
    configure_channels(win)
    win.ed_xlabel.setText("Time (ms)")
    win.ed_yllabel.setText("Current (A)")
    win.ed_yrlabel.setText("Control Signal (V)")
    win._rebuild_table()

    report = [
        "Scope Studio 03 UI Backtest",
        "===========================",
        f"CSV: {csv_path}",
        "",
    ]
    for idx in (1, 2, 3):
        assessment = apply_iteration(win, idx)
        win.refresh_plot()
        stats = win.compute_stats()
        app.processEvents()
        screenshot = os.path.join(ROOT, "backtests", f"iteration_{idx:02d}.png")
        win.grab().save(screenshot)
        report.append(f"Iteration {idx}")
        report.append("-" * 11)
        report.append(textwrap.fill(assessment, width=88))
        report.append(f"Screenshot: {screenshot}")
        report.append("Stats preview:")
        report.extend("  " + line for line in stats.splitlines()[:5])
        report.append("")

    out_report = os.path.join(ROOT, "backtests", "ui_backtest_report.txt")
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        f.write("\n")
    print(f"Wrote {out_report}")
    for idx in (1, 2, 3):
        print(os.path.join(ROOT, "backtests", f"iteration_{idx:02d}.png"))


if __name__ == "__main__":
    main()
