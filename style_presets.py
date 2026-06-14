"""
style_presets.py - journal-style plotting presets.

These presets are intentionally editable. They are starting points for young
scientists who need publication-standard figures without learning every
matplotlib rcParam first.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy


PALETTES = {
    "Wong / Nature": [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7",
        "#56B4E9", "#D55E00", "#F0E442", "#000000",
    ],
    "Nature Physics (NPG)": [
        "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
        "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
    ],
    "Nature Communications": [
        "#000000", "#D55E00", "#0072B2", "#009E73",
        "#CC79A7", "#56B4E9", "#E69F00", "#7F7F7F",
    ],
    "Science / AAAS": [
        "#1F77B4", "#D62728", "#2CA02C", "#9467BD",
        "#FF7F0E", "#17BECF", "#8C564B", "#7F7F7F",
    ],
    "APS / PRL": [
        "#000000", "#C44E52", "#4C72B0", "#55A868",
        "#8172B3", "#CCB974", "#64B5CD", "#8C8C8C",
    ],
    "Tokamak dual-axis": [
        "#000000", "#0072B2", "#D55E00", "#009E73",
        "#CC79A7", "#56B4E9", "#E69F00", "#7F7F7F",
    ],
}


BUILTIN_STYLE_PRESETS = {
    "Nature Physics": {
        "font_family": "Arial",
        "font_size": 7.0,
        "title_size": 8.0,
        "legend_font_size": 6.5,
        "line_width": 0.75,
        "axis_width": 0.5,
        "grid_alpha": 0.28,
        "palette": "Nature Physics (NPG)",
        "width_mm": 89.0,
        "height_mm": 60.0,
        "show_grid": False,
    },
    "Nature Communications": {
        "font_family": "Arial",
        "font_size": 7.0,
        "title_size": 8.0,
        "legend_font_size": 6.5,
        "line_width": 0.75,
        "axis_width": 0.5,
        "grid_alpha": 0.25,
        "palette": "Nature Communications",
        "width_mm": 89.0,
        "height_mm": 58.0,
        "show_grid": False,
    },
    "Science / AAAS": {
        "font_family": "Arial",
        "font_size": 8.0,
        "title_size": 9.0,
        "legend_font_size": 7.0,
        "line_width": 0.8,
        "axis_width": 0.6,
        "grid_alpha": 0.25,
        "palette": "Science / AAAS",
        "width_mm": 92.0,
        "height_mm": 62.0,
        "show_grid": False,
    },
    "APS / PRL": {
        "font_family": "Times New Roman",
        "font_size": 8.0,
        "title_size": 9.0,
        "legend_font_size": 7.0,
        "line_width": 0.85,
        "axis_width": 0.6,
        "grid_alpha": 0.22,
        "palette": "APS / PRL",
        "width_mm": 86.0,
        "height_mm": 58.0,
        "show_grid": False,
    },
    "Teaching / Slides": {
        "font_family": "Arial",
        "font_size": 11.0,
        "title_size": 13.0,
        "legend_font_size": 10.0,
        "line_width": 1.6,
        "axis_width": 1.0,
        "grid_alpha": 0.35,
        "palette": "Tokamak dual-axis",
        "width_mm": 160.0,
        "height_mm": 90.0,
        "show_grid": True,
    },
}


def _user_style_path(base_dir: str) -> str:
    return os.path.join(base_dir, "user_style_presets.json")


def load_style_presets(base_dir: str) -> dict[str, dict]:
    presets = deepcopy(BUILTIN_STYLE_PRESETS)
    try:
        with open(_user_style_path(base_dir), encoding="utf-8") as f:
            user_presets = json.load(f)
    except Exception:
        return presets
    if isinstance(user_presets, dict):
        for name, spec in user_presets.items():
            if isinstance(spec, dict):
                merged = deepcopy(BUILTIN_STYLE_PRESETS["Nature Physics"])
                merged.update(spec)
                presets[str(name)] = merged
    return presets


def save_user_style(base_dir: str, name: str, style: dict):
    path = _user_style_path(base_dir)
    try:
        with open(path, encoding="utf-8") as f:
            user_presets = json.load(f)
    except Exception:
        user_presets = {}
    user_presets[name] = dict(style)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(user_presets, f, indent=2)
        f.write("\n")


def style_palette(style: dict) -> list[str]:
    return PALETTES.get(style.get("palette", "Wong / Nature"),
                        PALETTES["Wong / Nature"])

