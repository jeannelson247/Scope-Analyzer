"""backend_api.py - Python<->JS bridge for the web Scope Analyzer app.

The browser is only the view. CSV loading, calibration, transforms,
anomaly scans, saturation estimates, and reconstructions are deterministic
Python/NumPy/SciPy operations. Source CSV files are never modified.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _resource_root() -> str:
    base = os.path.abspath(getattr(sys, "_MEIPASS", os.path.dirname(HERE)))
    candidates = [base]
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "Resources"))
        candidates.append(os.path.join(os.path.dirname(base), "Resources"))
    candidates.append(os.path.dirname(HERE))
    if not getattr(sys, "frozen", False):
        candidates.append(os.path.join(os.path.dirname(base), "Resources"))
    for candidate in dict.fromkeys(os.path.abspath(c) for c in candidates):
        if (os.path.exists(os.path.join(candidate, "examples")) or
                os.path.exists(os.path.join(candidate, "presets.json")) or
                os.path.exists(os.path.join(candidate, "scope_web", "index.html"))):
            return candidate
    return base


ROOT = _resource_root()
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from csv_loader import load_csv, minmax_decimate  # noqa: E402
from signal_tools import (  # noqa: E402
    FormulaError,
    amplitude_spectrum,
    dominant_frequency,
    evaluate_formula,
    lowpass,
    movmean,
    gradient,
    integrate,
)

PLOT_POINTS = 4000


def _examples_dir() -> Path:
    """Resolve the toolbox examples directory across dev and packaged runs.

    Order: explicit env override, the repo/bundle copy, then the first-run
    user copy under ~/Documents. Prefer a directory that has manifest.json.
    """
    cands: list[Path] = []
    env = os.environ.get("SCOPE_ANALYZER_EXAMPLES")
    if env:
        cands.append(Path(env))
    cands.append(Path(ROOT) / "examples" / "tool_benchmarks")
    cands.append(Path.home() / "Documents" / "Scope Analyzer" / "examples" / "tool_benchmarks")
    for c in cands:
        if (c / "manifest.json").exists():
            return c
    for c in cands:
        if c.is_dir():
            return c
    return cands[-1]


def _plain_float_list(values):
    return [float(v) for v in np.asarray(values, dtype=np.float64)]


def _parse_optional_float(value):
    if value in (None, "", "auto"):
        return None
    return float(value)


def _parse_windows(value) -> list[tuple[float, float]] | None:
    if value in (None, "", []):
        return None
    out: list[tuple[float, float]] = []
    if isinstance(value, str):
        for part in re.split(r"[;,]", value):
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", part)
            if len(nums) >= 2:
                a, b = float(nums[0]), float(nums[1])
                out.append(tuple(sorted((a, b))))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                a = item.get("start", item.get("lo", item.get("min")))
                b = item.get("end", item.get("hi", item.get("max")))
                if a is not None and b is not None:
                    out.append(tuple(sorted((float(a), float(b)))))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append(tuple(sorted((float(item[0]), float(item[1])))))
    return out or None


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _calibration_log_path() -> Path:
    explicit = os.environ.get("SCOPE_ANALYZER_CALIBRATION_LOG")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("SCOPE_ANALYZER_USER_DIR")
    root = Path(base).expanduser() if base else Path.home() / "Documents" / "Scope Analyzer"
    return root / "calibration_log.jsonl"


class Api:
    """Methods exposed to JavaScript. All returns are JSON-serializable."""

    def __init__(self):
        self._window = None
        self._loaded = None
        self._last_transforms: dict[str, dict[str, Any]] = {}

    def set_window(self, window):
        self._window = window

    # -- file open -----------------------------------------------------
    def pick_csv(self):
        if self._window is None:
            return {"ok": False, "error": "no window bound"}
        try:
            import webview
            sel = self._window.create_file_dialog(
                webview.OPEN_DIALOG, allow_multiple=False,
                file_types=("CSV files (*.csv;*.CSV)", "All files (*.*)"))
        except Exception as e:
            return {"ok": False, "error": f"dialog failed: {e}"}
        if not sel:
            return {"ok": False, "error": "cancelled"}
        return self.load_csv(sel[0])

    # -- helpers -------------------------------------------------------
    def _xcol(self):
        if self._loaded is None:
            return None
        return next((c for c in self._loaded.df.columns
                     if "time" in str(c).lower()), None)

    def _x(self) -> np.ndarray:
        if self._loaded is None:
            return np.zeros(0, dtype=np.float64)
        df = self._loaded.df
        xcol = self._xcol()
        if xcol is None:
            return np.arange(len(df), dtype=np.float64)
        return df[xcol].to_numpy(dtype=np.float64)

    def _column(self, column: str) -> np.ndarray:
        if column in self._last_transforms:
            return np.asarray(self._last_transforms[column]["values"],
                              dtype=np.float64)
        if self._loaded is None or column not in self._loaded.df.columns:
            raise KeyError("no such column / no data")
        return self._loaded.df[column].to_numpy(dtype=np.float64)

    def _window_mask(self, x, t_start=None, t_end=None):
        mask = np.ones(len(x), dtype=bool)
        if t_start not in (None, "", "auto"):
            mask &= x >= float(t_start)
        if t_end not in (None, "", "auto"):
            mask &= x <= float(t_end)
        return mask

    def _series(self, x, y):
        xs, ys = minmax_decimate(np.asarray(x, dtype=np.float64),
                                 np.asarray(y, dtype=np.float64),
                                 PLOT_POINTS)
        return {"x": _plain_float_list(xs), "y": _plain_float_list(ys)}

    def _shot_context(self) -> dict[str, Any]:
        if self._loaded is None:
            return {}
        return {
            "file_name": os.path.basename(self._loaded.path),
            "file_path": self._loaded.path,
            "n_rows": int(self._loaded.n_rows),
            "columns": [str(c) for c in self._loaded.df.columns],
            "x_column": str(self._xcol() or ""),
        }

    def save_calibration_log(self, entry: dict | None = None):
        """Append one calibration record to the user's persistent log.

        The log is intentionally separate from the source CSV. It records
        formulas, gains, fit slopes, and context for future reuse while keeping
        the raw data immutable.
        """
        entry = dict(entry or {})
        path = _calibration_log_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "kind": str(entry.pop("kind", "calibration")),
                "shot": self._shot_context(),
                "read_only_source_csv": True,
                **_json_safe(entry),
            }
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return {
                "ok": True,
                "path": str(path),
                "entry": payload,
                "text": f"Saved calibration record to {path}",
                "read_only": True,
            }
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "path": str(path), "read_only": True}

    def list_calibration_log(self, limit: int = 25):
        path = _calibration_log_path()
        if not path.exists():
            return {"ok": True, "entries": [], "path": str(path),
                    "text": f"No calibration log yet.\nNew records will be saved to:\n{path}",
                    "read_only": True}
        entries = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        entries.append({"kind": "unreadable", "raw": line})
            entries = entries[-max(int(limit), 1):][::-1]
            lines = [f"Calibration log: {path}", ""]
            for e in entries:
                shot = e.get("shot", {}) or {}
                bits = [
                    str(e.get("timestamp", "")),
                    str(e.get("kind", "calibration")),
                    str(e.get("label") or e.get("preset_name") or e.get("source") or ""),
                ]
                if shot.get("file_name"):
                    bits.append(f"file={shot.get('file_name')}")
                if e.get("formula"):
                    bits.append(f"formula={e.get('formula')}")
                if e.get("slope") is not None:
                    bits.append(f"slope={float(e.get('slope')):.6g}")
                if e.get("gain") is not None:
                    bits.append(f"gain={e.get('gain')}")
                if e.get("offset") is not None:
                    bits.append(f"offset={e.get('offset')}")
                lines.append(" - " + " | ".join(b for b in bits if b))
            return {"ok": True, "entries": entries, "path": str(path),
                    "text": "\n".join(lines), "read_only": True}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "path": str(path), "read_only": True}

    def _stats_from_arrays(self, column, x, y, t_start=None, t_end=None):
        mask = self._window_mask(x, t_start, t_end)
        y2 = np.asarray(y, dtype=np.float64)[mask]
        x2 = np.asarray(x, dtype=np.float64)[mask]
        finite = y2[np.isfinite(y2)]
        window = [float(np.nanmin(x2)), float(np.nanmax(x2))] if x2.size else None
        if finite.size == 0:
            return {"ok": False, "error": "no finite samples in range",
                    "column": column, "window": window}
        return {"ok": True, "column": column,
                "n": int(y2.size), "n_finite": int(finite.size),
                "min": float(np.min(finite)), "max": float(np.max(finite)),
                "mean": float(np.mean(finite)), "std": float(np.std(finite)),
                "median": float(np.median(finite)),
                "p5": float(np.percentile(finite, 5)),
                "p95": float(np.percentile(finite, 95)),
                "rms": float(np.sqrt(np.mean(np.square(finite)))),
                "window": window, "read_only": True}

    # -- data ----------------------------------------------------------
    def load_csv(self, path: str):
        try:
            ld = load_csv(path)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self._loaded = ld
        self._last_transforms.clear()
        df = ld.df
        cols = [str(c) for c in df.columns]
        xcol = next((c for c in cols if "time" in c.lower()), None)
        x = (df[xcol].to_numpy(dtype=np.float64) if xcol else
             np.arange(len(df), dtype=np.float64))
        series = {}
        for c in cols:
            if c == xcol:
                continue
            y = df[c].to_numpy(dtype=np.float64)
            series[c] = self._series(x, y)
        return {
            "ok": True,
            "path": path,
            "name": os.path.basename(path),
            "columns": cols,
            "x_col": xcol,
            "y_cols": [c for c in cols if c != xcol],
            "units": {str(k): str(v) for k, v in (ld.units or {}).items()},
            "n_rows": int(ld.n_rows),
            "series": series,
            "read_only": True,
        }

    def list_presets(self):
        path = os.path.join(ROOT, "presets.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception as e:
            return {"ok": False, "error": str(e), "presets": []}
        presets = []
        for name, spec in raw.items():
            presets.append({
                "name": str(name),
                "gain": float(spec.get("gain", 1.0)),
                "offset": float(spec.get("offset", 0.0)),
                "unit": str(spec.get("unit", "")),
                "formula": str(spec.get("formula", "x") or "x"),
            })
        return {"ok": True, "presets": presets}

    def list_examples(self):
        """List bundled toolbox example datasets from manifest.json so the UI
        can offer a one-click guided tour for first-time users."""
        d = _examples_dir()
        try:
            data = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        except Exception as e:
            return {"ok": False, "error": f"no example manifest: {e}", "examples": []}
        items = []
        for ds in data.get("datasets", []):
            f = str(ds.get("file", ""))
            if not f or not (d / f).exists():
                continue
            items.append({
                "file": f,
                "id": str(ds.get("id", "")),
                "title": str(ds.get("title", f)),
                "tools": list(ds.get("tools", [])),
            })
        return {"ok": True, "dir": str(d), "examples": items}

    def load_example(self, file: str):
        """Load one bundled example CSV by name (read-only; no path traversal)."""
        d = _examples_dir()
        name = os.path.basename(str(file))
        path = d / name
        if not path.exists():
            return {"ok": False, "error": f"example not found: {name}"}
        return self.load_csv(str(path))

    def list_tools(self):
        return {"ok": True, "tools": [
            {"id": "stats", "group": "Summary", "name": "Statistics"},
            {"id": "quality", "group": "Summary", "name": "CSV quality report"},
            {"id": "formula", "group": "Transforms", "name": "Formula / calibration preset"},
            {"id": "lowpass", "group": "Transforms", "name": "Low-pass filter"},
            {"id": "movmean", "group": "Transforms", "name": "Moving average"},
            {"id": "gradient", "group": "Transforms", "name": "Derivative / dI/dt"},
            {"id": "integrate", "group": "Transforms", "name": "Integral"},
            {"id": "fft", "group": "Frequency", "name": "FFT / dominant frequency"},
            {"id": "anomaly", "group": "Diagnostics", "name": "Anomaly scan"},
            {"id": "saturation", "group": "Diagnostics", "name": "Saturation estimate"},
            {"id": "rlc", "group": "Reconstruction", "name": "Censored RLC reconstruction"},
            {"id": "calibration", "group": "Calibration", "name": "Forced-origin reference gain"},
            {"id": "cal_log", "group": "Calibration", "name": "Calibration log"},
            {"id": "pipeline", "group": "Workflow", "name": "Analyze shot pipeline"},
            {"id": "help", "group": "Help", "name": "Toolbox FAQ and examples"},
        ]}

    def apply_channel(self, column: str, formula: str = "x",
                      gain: float = 1.0, offset: float = 0.0,
                      label: str = "", unit: str = "",
                      preset_name: str = "", save_log: bool = False):
        """Return a transformed/decimated display trace. CSV untouched."""
        try:
            x = self._x()
            raw = self._column(column)
            y = evaluate_formula(formula or "x", raw, x, x * 1000.0)
            y = y * float(gain) + float(offset)
            name = label.strip() or f"{column} derived"
            self._last_transforms[name] = {
                "source": column, "formula": formula or "x",
                "gain": float(gain), "offset": float(offset),
                "unit": unit or "", "values": y.copy(),
            }
            result = {"ok": True, "source": column, "label": name,
                      "unit": unit or "", "formula": formula or "x",
                      "gain": float(gain), "offset": float(offset),
                      "preset_name": preset_name or "",
                      "series": self._series(x, y),
                      "stats": self._stats_from_arrays(name, x, y),
                      "read_only": True}
            if save_log:
                result["log"] = self.save_calibration_log({
                    "kind": "display_formula",
                    "source": column,
                    "label": name,
                    "preset_name": preset_name or "",
                    "formula": formula or "x",
                    "gain": float(gain),
                    "offset": float(offset),
                    "unit": unit or "",
                })
            return result
        except (KeyError, FormulaError, ValueError) as e:
            return {"ok": False, "error": str(e), "read_only": True}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "read_only": True}

    def column_stats(self, column: str, t_start=None, t_end=None):
        if self._loaded is None:
            return {"ok": False, "error": "no such column / no data"}
        if column not in self._loaded.df.columns and column not in self._last_transforms:
            return {"ok": False, "error": "no such column / no data"}
        return self._stats_from_arrays(column, self._x(), self._column(column),
                                       t_start, t_end)

    def run_tool(self, tool_id: str, params: dict | None = None):
        params = params or {}
        tool_id = str(tool_id or "").lower()
        if tool_id == "analyze":
            tool_id = "pipeline"
        if self._loaded is None:
            return {"ok": False, "error": "load a CSV first"}
        try:
            x = self._x()
            col = params.get("column") or params.get("source")
            if not col:
                ycols = [c for c in self._loaded.df.columns if c != self._xcol()]
                col = str(ycols[0]) if ycols else ""

            if tool_id in ("stats", "statistics"):
                return self.column_stats(str(col), params.get("t_start"), params.get("t_end"))

            if tool_id == "quality":
                from data_quality import quality_report
                rep = quality_report(self._loaded)
                return {"ok": True, "text": rep.one_line(),
                        "status": rep.status, "read_only": True}

            if tool_id in ("formula", "apply_channel"):
                return self.apply_channel(
                    str(col), params.get("formula", "x"),
                    float(params.get("gain", 1.0) or 1.0),
                    float(params.get("offset", 0.0) or 0.0),
                    str(params.get("label", "") or ""),
                    str(params.get("unit", "") or ""),
                    str(params.get("preset_name", "") or ""),
                    _truthy(params.get("save_log")))

            if tool_id == "lowpass":
                cutoff = float(params.get("cutoff_hz", 10000.0) or 10000.0)
                y = lowpass(self._column(str(col)), x, cutoff)
                label = params.get("label") or f"{col} low-pass {cutoff:g} Hz"
                return {"ok": True, "label": label, "series": self._series(x, y),
                        "text": f"Applied deterministic low-pass filter at {cutoff:g} Hz to {col}.",
                        "read_only": True}

            if tool_id == "movmean":
                window = int(float(params.get("window", 101) or 101))
                y = movmean(self._column(str(col)), window)
                label = params.get("label") or f"{col} movmean {window}"
                return {"ok": True, "label": label, "series": self._series(x, y),
                        "text": f"Applied moving average window={window} samples to {col}.",
                        "read_only": True}

            if tool_id == "gradient":
                y = gradient(self._column(str(col)), x)
                label = params.get("label") or f"d/dt {col}"
                return {"ok": True, "label": label, "series": self._series(x, y),
                        "text": f"Computed derivative/dI-dt for {col} using NumPy gradient.",
                        "read_only": True}

            if tool_id == "integrate":
                y = integrate(self._column(str(col)), x)
                label = params.get("label") or f"integral {col}"
                return {"ok": True, "label": label, "series": self._series(x, y),
                        "text": f"Computed cumulative trapezoid-style integral for {col}.",
                        "read_only": True}

            if tool_id == "fft":
                y = self._column(str(col))
                mask = self._window_mask(x, params.get("t_start"), params.get("t_end"))
                xs, ys = x[mask], y[mask]
                if xs.size < 8:
                    return {"ok": False, "error": "FFT needs at least 8 samples"}
                dt = float(np.median(np.diff(xs)))
                freq, amp = amplitude_spectrum(ys - np.nanmean(ys), dt)
                dom = dominant_frequency(ys - np.nanmean(ys), dt,
                                         float(params.get("f_min", 0.0) or 0.0))
                return {"ok": True, "text": f"{col}: dominant frequency {dom:.6g} Hz over {xs.size:,} samples.",
                        "dominant_frequency_hz": float(dom),
                        "series": {"x": _plain_float_list(freq[:PLOT_POINTS]),
                                   "y": _plain_float_list(amp[:PLOT_POINTS])},
                        "read_only": True}

            if tool_id == "anomaly":
                import detect_anomalies as da
                y = self._column(str(col))
                mask = self._window_mask(x, params.get("t_start"), params.get("t_end"))
                if mask.sum() < 16:
                    return {"ok": False, "error": "not enough samples in window"}
                rep = da.detect(x[mask], {str(col): y[mask]},
                                threshold_sigma=float(params.get("threshold_sigma", 6.0) or 6.0),
                                crest_limit=float(params.get("crest_limit", 5.0) or 5.0),
                                x_unit=str(params.get("x_unit", "s")))
                return {"ok": True, "text": rep.text(), "read_only": True}

            if tool_id == "saturation":
                from saturation_recovery import estimate_true_current
                y = self._column(str(col))
                ref_name = params.get("ref_channel") or params.get("reference")
                yref = self._column(str(ref_name)) if ref_name else None
                sat = _parse_optional_float(params.get("sat_level"))
                cal_window = None
                if params.get("cal_start") not in (None, "") or params.get("cal_end") not in (None, ""):
                    cal_window = (float(params.get("cal_start", x[0])),
                                  float(params.get("cal_end", x[-1])))
                rep = estimate_true_current(x, y, label=str(col),
                                            y_ref=yref,
                                            ref_label=str(ref_name or ""),
                                            cal_window=cal_window,
                                            sat_level=sat)
                return {"ok": True, "text": rep.text,
                        "overlay": rep.overlay, "read_only": True}

            if tool_id == "rlc":
                from rlc_reconstruct import fit_rlc
                y = self._column(str(col))
                ref_name = params.get("ref_channel") or params.get("reference")
                yref = self._column(str(ref_name)) if ref_name else None
                sat = _parse_optional_float(params.get("sat_level"))
                t_window = None
                if params.get("t_start") not in (None, "") or params.get("t_end") not in (None, ""):
                    t_window = (float(params.get("t_start", x[0])),
                                float(params.get("t_end", x[-1])))
                ref_window = None
                if yref is not None and params.get("ref_end") not in (None, ""):
                    ref_window = (float(params.get("ref_start", x[0])),
                                  float(params.get("ref_end")))
                rep = fit_rlc(x, y, sat_level=sat, label=str(col),
                              t_window=t_window, y_ref=yref,
                              ref_window=ref_window,
                              ref_label=str(ref_name or ""),
                              trusted_windows=_parse_windows(params.get("trusted_windows")))
                return {"ok": rep.ok, "text": rep.text,
                        "overlay": rep.curve, "params": rep.params,
                        "read_only": True}

            if tool_id == "calibration":
                from calibration import fit_forced_origin_gain, format_gain_fit
                source = str(params.get("source") or col)
                ref = str(params.get("ref_channel") or params.get("reference") or "")
                if not ref:
                    return {"ok": False, "error": "choose a reference channel"}
                lo = float(params.get("t_start", np.nanmin(x)))
                hi = float(params.get("t_end", np.nanmax(x)))
                source_formula = str(params.get("source_formula") or "x")
                source_gain = float(params.get("source_gain", 1.0) or 1.0)
                source_offset = float(params.get("source_offset", 0.0) or 0.0)
                y_source = evaluate_formula(source_formula, self._column(source),
                                            x, x * 1000.0)
                y_source = y_source * source_gain + source_offset
                source_label = str(params.get("source_preset_name") or source)
                res = fit_forced_origin_gain(x, y_source, self._column(ref), lo, hi)
                text = format_gain_fit(res, source_label, "time")
                result = {"ok": True,
                          "text": text,
                          "source": source,
                          "reference": ref,
                          "source_preset_name": source_label,
                          "source_formula": source_formula,
                          "source_gain": source_gain,
                          "source_offset": source_offset,
                          "slope": float(res.slope),
                          "ci": [float(res.ci_lo), float(res.ci_hi)],
                          "r2": float(res.r2),
                          "n_samples": int(res.n_samples),
                          "window": [float(res.window[0]), float(res.window[1])],
                          "read_only": True}
                if _truthy(params.get("save_log")):
                    result["log"] = self.save_calibration_log({
                        "kind": "reference_fit",
                        "source": source,
                        "reference": ref,
                        "source_preset_name": source_label,
                        "formula": source_formula,
                        "gain": source_gain,
                        "offset": source_offset,
                        "slope": float(res.slope),
                        "ci_lo": float(res.ci_lo),
                        "ci_hi": float(res.ci_hi),
                        "r2": float(res.r2),
                        "n_samples": int(res.n_samples),
                        "window": [float(res.window[0]), float(res.window[1])],
                    })
                return result

            if tool_id == "cal_log":
                return self.list_calibration_log(int(float(params.get("limit", 25) or 25)))

            if tool_id == "pipeline":
                y = self._column(str(col))
                span = float(np.nanmax(x) - np.nanmin(x)) if x.size else 0.0
                # Native app default: Jean's four-BBCM rig clips near 6 kA.
                # Apply it only when the displayed trace is in the kA-scale
                # regime, so generic teaching datasets are not overfit.
                sat_default = 6000.0 if float(np.nanmax(np.abs(y))) > 3000.0 else ""
                sat_level = params.get("sat_level", sat_default)
                ref_end = params.get("ref_end", 0.005 if span < 10.0 else 5.0)
                parts = [self.column_stats(str(col))]
                parts.append(self.run_tool("anomaly", {"column": col}))
                parts.append(self.run_tool("saturation", {"column": col,
                                                           "sat_level": sat_level}))
                parts.append(self.run_tool("rlc", {"column": col,
                                                    "sat_level": sat_level,
                                                    "ref_end": ref_end,
                                                    "trusted_windows": params.get("trusted_windows", "")}))
                text = "\n\n".join(p.get("text") or json.dumps(p) for p in parts)
                overlay = next((p.get("overlay") for p in reversed(parts)
                                if p.get("ok") and p.get("overlay")), None)
                return {"ok": True, "text": text, "overlay": overlay,
                        "read_only": True}

            return {"ok": False, "error": f"unknown tool: {tool_id}"}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "read_only": True}


    def toolbox_help(self):
        """Return bundled no-LLM toolbox guidance for the Lite help panel."""
        candidates = [
            os.path.join(ROOT, "docs", "TOOLBOX_FAQ.md"),
            os.path.join(os.path.dirname(ROOT), "docs", "TOOLBOX_FAQ.md"),
        ]
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    text = handle.read()
                return {"ok": True, "text": text, "path": path,
                        "read_only": True}
            except OSError:
                continue
        return {"ok": True, "read_only": True, "text": (
            "Scope Analyzer Lite toolbox help\n\n"
            "Open Tools & libraries for deterministic tools: Statistics, CSV "
            "quality, Formula, Low-pass, Moving average, Derivative/dI/dt, "
            "Integral, FFT, Anomaly scan, Saturation estimate, RLC "
            "reconstruction, Reference calibration, and Analyze shot.\n\n"
            "Benchmark examples live in examples/tool_benchmarks/. Source CSV "
            "files are read-only; transforms and overlays are in memory."
        )}

    # -- lightweight placeholders -------------------------------------
    def list_models(self):
        return {"ok": True, "models": [], "note": "model selection is optional"}

    def chat(self, prompt: str, backend: str = "mlx", model: str = ""):
        return {"ok": False, "error": "chat bridge not wired yet"}


def selftest() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    candidates = [os.path.join(root, "Claidev2", "uploads", "T0026.CSV")]
    ex = os.path.join(root, "examples")
    if os.path.isdir(ex):
        for d, _, fs in os.walk(ex):
            for f in fs:
                if f.lower().endswith(".csv"):
                    candidates.append(os.path.join(d, f))
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        print("SELFTEST SKIP: no sample CSV found")
        return 0
    api = Api()
    r = api.load_csv(path)
    fails = []
    if not r.get("ok"):
        fails.append(f"load failed: {r.get('error')}")
    else:
        if not r["columns"]:
            fails.append("no columns")
        if not r["series"]:
            fails.append("no series")
        for c, s in r["series"].items():
            if len(s["x"]) != len(s["y"]):
                fails.append(f"{c}: x/y length mismatch")
            if len(s["y"]) > PLOT_POINTS + 4:
                fails.append(f"{c}: not decimated ({len(s['y'])})")
        st = api.column_stats(r["y_cols"][0]) if r["y_cols"] else {"ok": False}
        if not st.get("ok"):
            fails.append("column_stats failed")
        pr = api.list_presets()
        if not pr.get("ok"):
            fails.append("list_presets failed")
    print(f"loaded: {r.get('name')}  rows={r.get('n_rows')}  "
          f"cols={r.get('columns')}  x={r.get('x_col')}  "
          f"series={list(r.get('series', {}).keys())}")
    if fails:
        print("BACKEND-API SELFTEST: FAIL")
        for f in fails:
            print("  [FAIL]", f)
        return 1
    print("BACKEND-API SELFTEST: PASS (real CSV -> decimated series + stats + presets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
