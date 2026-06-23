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
import base64
import subprocess
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


def _generate_examples_if_missing(target: Path) -> None:
    """Create the synthetic no-LLM examples when a checkout/bundle lacks them.

    The examples are deterministic teaching data, not user measurements. They
    are safe to regenerate because they never touch a loaded source CSV.
    """
    if (target / "manifest.json").exists():
        return
    try:
        from scripts.generate_lite_toolbox_examples import make_examples

        make_examples(target)
    except Exception:
        # list_examples() will return the original "no manifest" message.
        return


def _generate_stress_examples_if_missing(target: Path) -> None:
    """Create the advanced stress examples when absent.

    These are generated separately from the beginner examples so the friendly
    guided tour stays small, while developers/users still get tougher cases for
    fault-finding.
    """
    if (target / "manifest.json").exists():
        return
    try:
        from scripts.generate_lite_stress_examples import make_stress_examples

        make_stress_examples(target)
    except Exception:
        return


def _examples_dir() -> Path:
    """Resolve the toolbox examples directory across dev and packaged runs.

    Order: explicit env override, the repo/bundle copy, then the first-run
    user copy under ~/Documents. Prefer a directory that has manifest.json.
    """
    cands: list[Path] = []
    env = os.environ.get("SCOPE_ANALYZER_EXAMPLES")
    if env:
        env_path = Path(env)
        _generate_examples_if_missing(env_path)
        if (env_path / "manifest.json").exists():
            return env_path
        cands.append(env_path)
    bundled = Path(ROOT) / "examples" / "tool_benchmarks"
    user_copy = Path.home() / "Documents" / "Scope Analyzer" / "examples" / "tool_benchmarks"
    cands.append(bundled)
    cands.append(user_copy)
    for c in cands:
        if (c / "manifest.json").exists():
            return c
    # If a fresh checkout or app bundle has no generated examples, create them
    # in a writable location so the Examples menu is never a dead end.
    generate_target = Path(env) if env else (user_copy if getattr(sys, "frozen", False) else bundled)
    _generate_examples_if_missing(generate_target)
    if (generate_target / "manifest.json").exists():
        return generate_target
    for c in cands:
        if c.is_dir():
            return c
    return cands[-1]


def _stress_examples_dir() -> Path:
    """Resolve the advanced stress-test examples directory."""
    cands: list[Path] = []
    env = os.environ.get("SCOPE_ANALYZER_STRESS_EXAMPLES")
    if env:
        env_path = Path(env)
        _generate_stress_examples_if_missing(env_path)
        if (env_path / "manifest.json").exists():
            return env_path
        cands.append(env_path)
    bundled = Path(ROOT) / "examples" / "tool_stress"
    user_copy = Path.home() / "Documents" / "Scope Analyzer" / "examples" / "tool_stress"
    cands.append(bundled)
    cands.append(user_copy)
    for c in cands:
        if (c / "manifest.json").exists():
            return c
    generate_target = Path(env) if env else (user_copy if getattr(sys, "frozen", False) else bundled)
    _generate_stress_examples_if_missing(generate_target)
    if (generate_target / "manifest.json").exists():
        return generate_target
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


def _bool_param(params: dict, key: str, default: bool = True) -> bool:
    if key not in params:
        return default
    return _truthy(params.get(key))


def _physical_rlc_params(params: dict) -> dict[str, float | None]:
    """Parse optional physical R/L/C/V0 hints in SI units.

    The UI sends R in ohms, L in henries, and C in farads. A few aliases are
    accepted so scripts can call the bridge directly without caring about the
    exact frontend field names.
    """
    R = _parse_optional_float(params.get("resistance_ohm", params.get("r_ohm")))
    L = _parse_optional_float(params.get("inductance_h", params.get("l_h")))
    if L is None:
        L_uh = _parse_optional_float(params.get("inductance_uh"))
        L = None if L_uh is None else L_uh * 1e-6
    C = _parse_optional_float(params.get("capacitance_f", params.get("c_f")))
    if C is None:
        C_mf = _parse_optional_float(params.get("capacitance_mf"))
        C = None if C_mf is None else C_mf * 1e-3
    V0 = _parse_optional_float(params.get(
        "charging_voltage_v",
        params.get("initial_voltage_v",
                   params.get("v0_v", params.get("capacitor_voltage_v")))))
    prior = _parse_optional_float(params.get("physical_prior_weight"))
    return {
        "resistance_ohm": R,
        "inductance_h": L,
        "capacitance_f": C,
        "charging_voltage_v": V0,
        "physical_prior_weight": 0.0 if prior is None else prior,
    }


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


def _unique_column_name(name: str, existing: set[str]) -> str:
    base = str(name or "column").strip() or "column"
    out = base
    i = 2
    while out in existing:
        out = f"{base}_{i}"
        i += 1
    existing.add(out)
    return out


def _calibration_log_path() -> Path:
    explicit = os.environ.get("SCOPE_ANALYZER_CALIBRATION_LOG")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("SCOPE_ANALYZER_USER_DIR")
    root = Path(base).expanduser() if base else Path.home() / "Documents" / "Scope Analyzer"
    return root / "calibration_log.jsonl"


def _first_meta_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def _delimiter_name(delim: str) -> str:
    return {"\t": "tab", ",": "comma", ";": "semicolon"}.get(delim, repr(delim))


def _quality_payload(rep) -> dict[str, Any]:
    return {
        "status": rep.status,
        "one_line": rep.one_line(),
        "issues": list(rep.issues),
        "n_rows": int(rep.n_rows),
        "n_columns": int(rep.n_columns),
        "x_column": rep.x_column,
        "sample_interval_s": rep.sample_interval_s,
        "sample_rate_hz": rep.sample_rate_hz,
        "nonfinite_time_count": int(rep.nonfinite_time_count),
        "duplicate_timestamp_count": int(rep.duplicate_timestamp_count),
        "backwards_timestamp_count": int(rep.backwards_timestamp_count),
        "large_gap_count": int(rep.large_gap_count),
        "max_gap_s": rep.max_gap_s,
        "total_nonfinite_values": int(rep.total_nonfinite_values),
        "nonfinite_by_column": dict(rep.nonfinite_by_column),
        "flatline_runs_by_column": dict(getattr(rep, "flatline_runs_by_column", {})),
        "longest_flatline_by_column": _json_safe(
            getattr(rep, "longest_flatline_by_column", {})),
    }


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    text = str(data_url or "")
    if not text.startswith("data:") or "," not in text:
        raise ValueError("expected a data URL")
    header, payload = text.split(",", 1)
    mime = header[5:].split(";", 1)[0].strip().lower()
    if ";base64" not in header.lower():
        raise ValueError("expected base64 image data")
    return base64.b64decode(payload), mime


def _copy_text_to_clipboard_native(text: str) -> tuple[bool, str]:
    """Copy text through native macOS APIs, then pbcopy as a fallback."""
    try:
        from AppKit import NSPasteboard, NSPasteboardTypeString

        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        ok = bool(pb.setString_forType_(str(text), NSPasteboardTypeString))
        if ok:
            return True, "native pasteboard"
    except Exception:
        pass
    try:
        subprocess.run(["pbcopy"], input=str(text).encode("utf-8"),
                       check=True, timeout=5)
        return True, "pbcopy"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _copy_image_to_clipboard_native(data: bytes, mime: str) -> tuple[bool, str]:
    """Copy PNG/JPEG bytes through the macOS pasteboard.

    Browser clipboard writes are commonly blocked inside local file/webview
    contexts. The packaged Lite app can use AppKit directly, which makes
    right-click Copy PNG/JPG behave like a normal native app.
    """
    try:
        from AppKit import NSPasteboard
        from Foundation import NSData

        uti = "public.png" if "png" in mime else "public.jpeg"
        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        nsdata = NSData.dataWithBytes_length_(data, len(data))
        ok = bool(pb.setData_forType_(nsdata, uti))
        if ok:
            return True, "native pasteboard"
        return False, "pasteboard rejected image data"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _import_report_payload(ld, quality, path: str, xcol: str | None,
                           ycols: list[str]) -> dict[str, Any]:
    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)
    units = {str(k): str(v) for k, v in (ld.units or {}).items()}
    unit_summary = ", ".join(f"{k}: {v}" for k, v in list(units.items())[:6])
    if len(units) > 6:
        unit_summary += f", +{len(units) - 6} more"
    model = _first_meta_value((ld.meta or {}).get("Model"))
    source = ", ".join(v for v in (ld.meta or {}).get("Source", []) if str(v).strip())
    q = _quality_payload(quality)
    next_step = (
        "QC issue detected: open Tools & libraries -> CSV quality report before deeper analysis."
        if quality.status != "ok"
        else "Ready: choose a channel, apply any calibration/preset, then run Statistics, FFT, derivative, integral, or reconstruction."
    )
    lines = [
        "Import report",
        "=============",
        f"File: {os.path.basename(path)} ({size_mb:.2f} MB)",
        f"Read-only source CSV: yes",
        f"Detected scope/model: {model or 'unknown'}",
        f"Delimiter: {_delimiter_name(ld.delimiter)}   skipped header rows: {ld.skiprows}",
        f"Rows x columns: {ld.n_rows:,} x {len(ld.columns)}",
        f"X/time column: {xcol or 'sample index'}",
        f"Signal columns: {len(ycols)}",
        f"Units: {unit_summary or 'none detected'}",
        f"Source row: {source or 'none detected'}",
        f"Quality: {q['one_line']}",
        "",
        f"Suggested next step: {next_step}",
    ]
    return {
        "text": "\n".join(lines),
        "file_name": os.path.basename(path),
        "file_size_bytes": int(size_bytes),
        "file_size_mb": float(size_mb),
        "read_only": True,
        "delimiter": ld.delimiter,
        "delimiter_name": _delimiter_name(ld.delimiter),
        "skiprows": int(ld.skiprows),
        "scope_model": model,
        "source_channels": source,
        "rows": int(ld.n_rows),
        "columns": [str(c) for c in ld.columns],
        "x_column": xcol or "",
        "signal_columns": list(ycols),
        "units_detected": units,
        "quality": q,
        "suggested_next_step": next_step,
    }


EXAMPLE_GUIDES: dict[str, dict[str, Any]] = {
    "01_clean_rl_pulse.csv": {
        "tool": "stats",
        "column": "Current_A",
        "why": "Start with caption-ready statistics on a clean pulse before trying derivative, integral, or RLC overlays.",
        "expected": "Peak current should be near 4.2 kA and the source CSV remains read-only.",
    },
    "02_bbcm_clipped_6ka.csv": {
        "tool": "rlc",
        "column": "BBCM_A",
        "why": "Recover a hidden peak from a 6 kA clipped BBCM trace using the early Pearson reference and later clean BBCM window.",
        "params": {
            "sat_level": 6000,
            "ref_channel": "Pearson_A",
            "ref_start": 0,
            "ref_end": 0.010,
            "trusted_windows": "0:0.005, 0.040:0.150",
        },
        "expected": "The reconstructed peak should land around 7.4 kA and draw a dashed model-estimate overlay.",
    },
    "03_lowpass_ringing.csv": {
        "tool": "lowpass",
        "column": "Current_noisy_A",
        "why": "Apply a 15 kHz low-pass filter to suppress 150 kHz ringing while preserving the current envelope.",
        "params": {"cutoff_hz": 15000, "label": "Current_noisy_A LP 15 kHz"},
        "expected": "A smoother derived trace appears; the original noisy channel is still available.",
    },
    "04_fft_two_tone.csv": {
        "tool": "fft",
        "column": "Signal_V",
        "why": "Use FFT to identify the dominant tone in a mixed-frequency voltage signal.",
        "params": {"f_min": 1000},
        "expected": "Dominant frequency should be close to 20 kHz.",
    },
    "05_calibration_pair.csv": {
        "tool": "formula",
        "column": "Sensor_V",
        "why": "Convert a centered sensor voltage to current with the same style of calibration formula used for current monitors.",
        "params": {
            "formula": "(x-2.5)*750",
            "label": "Sensor_V_to_A",
            "unit": "A",
            "gain": 1,
            "offset": 0,
        },
        "expected": "The new derived trace should overlap the reference current scale.",
    },
    "06_didt_voltage_166uH.csv": {
        "tool": "gradient",
        "column": "Current_A",
        "why": "Compute dI/dt and compare it to the supplied inductive-voltage channel using L = 166 uH.",
        "params": {"label": "dI/dt Current_A"},
        "expected": "The derivative trace explains the relationship V_L = L*dI/dt.",
    },
    "07_charge_integral.csv": {
        "tool": "integrate",
        "column": "Current_A",
        "why": "Integrate current over time to estimate delivered charge.",
        "params": {"label": "Charge_C"},
        "expected": "The final integral should be about 5 C.",
    },
    "08_moving_average_noise.csv": {
        "tool": "movmean",
        "column": "Noisy_current_A",
        "why": "Use a moving average when you want a simple smoothing pass rather than a frequency-domain filter.",
        "params": {"window": 101, "label": "Noisy_current_A movmean 101"},
        "expected": "A smoother plateau trace appears without changing the raw channel.",
    },
    "09_spikes_anomalies.csv": {
        "tool": "anomaly",
        "column": "Current_A",
        "why": "Detect sparse spikes that would be easy to miss when zoomed out.",
        "params": {"threshold_sigma": 5},
        "expected": "The report should flag injected spike/crest-factor events.",
    },
    "10_quality_gap_nan_duplicate.csv": {
        "tool": "quality",
        "column": "Signal_V",
        "why": "Run this before trusting analysis when a CSV may have timing gaps, duplicate samples, or NaNs.",
        "expected": "The report should flag timing and nonfinite-data issues.",
    },
    "11_baseline_offset.csv": {
        "tool": "formula",
        "column": "Raw_offset_A",
        "why": "Subtract a pre-trigger baseline using the formula helper baseline(x,t,end).",
        "params": {
            "formula": "baseline(x,t,-0.001)",
            "label": "Baseline_corrected_A",
            "unit": "A",
            "gain": 1,
            "offset": 0,
        },
        "expected": "The pre-trigger region should move close to zero.",
    },
    "12_soft_saturation.csv": {
        "tool": "rlc",
        "column": "Soft_BBCM_A",
        "why": "Treat compressed high-current samples as constrained measurements and fit the clean windows.",
        "params": {
            "sat_level": 6000,
            "trusted_windows": "0:0.005, 0.050:0.140",
        },
        "expected": "The overlay estimates the hidden current while noting the model assumptions.",
    },
    "13_module_balance.csv": {
        "tool": "stats",
        "column": "Module3_A",
        "why": "Use statistics to compare modules and find imbalance without needing an LLM.",
        "expected": "Module3 should show the largest peak in this synthetic balance example.",
    },
    "14_negative_pulse.csv": {
        "tool": "rlc",
        "column": "Negative_current_A",
        "why": "Check that reconstruction and statistics preserve sign for negative-polarity shots.",
        "params": {"t_start": 0.0, "t_end": 0.115},
        "expected": "The fitted peak should remain negative rather than being silently sign-flipped.",
    },
    "15_vi_didt_166uH.csv": {
        "tool": "gradient",
        "column": "Current_A",
        "why": "Connect drive voltage, current rise, and dI/dt for the 166 uH teaching model.",
        "params": {"label": "dI/dt Current_A"},
        "expected": "The derived dI/dt is consistent with the supplied L*dI/dt voltage scale.",
    },
}


class Api:
    """Methods exposed to JavaScript. All returns are JSON-serializable."""

    def __init__(self):
        self._window = None
        self._loaded = None
        self._last_transforms: dict[str, dict[str, Any]] = {}

    def set_window(self, window):
        self._window = window

    # -- native clipboard bridge --------------------------------------
    def copy_text_to_clipboard(self, text: str):
        """Copy text/SVG through the native app when browser clipboard is blocked."""
        ok, detail = _copy_text_to_clipboard_native(str(text or ""))
        return {"ok": ok, "detail": detail, "read_only": True}

    def copy_image_to_clipboard(self, data_url: str):
        """Copy PNG/JPG data URL through the native macOS pasteboard."""
        try:
            data, mime = _decode_data_url(data_url)
            if mime not in {"image/png", "image/jpeg", "image/jpg"}:
                return {"ok": False, "error": f"unsupported clipboard image type: {mime}",
                        "read_only": True}
            ok, detail = _copy_image_to_clipboard_native(data, mime)
            return {"ok": ok, "detail": detail, "mime": mime, "read_only": True}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "read_only": True}

    # -- file open -----------------------------------------------------
    def pick_csv(self):
        if self._window is None:
            return {"ok": False, "error": "no window bound"}
        try:
            import webview
            open_dialog = webview.OPEN_DIALOG
        except Exception as e:
            # Unit tests and some lightweight embeddings provide a window shim
            # without importing pywebview. The concrete value is ignored by
            # those shims; real packaged runs import pywebview above.
            open_dialog = 0
        # pywebview's filter description allows only [word/space] chars and the
        # parser varies by version; never hard-fail the user over a filter.
        file_types = (
            "Scope data (*.csv;*.CSV;*.txt;*.TXT;*.tsv;*.TSV)",
            "CSV files (*.csv;*.CSV)",
            "Text and TSV files (*.txt;*.TXT;*.tsv;*.TSV)",
            "All files (*.*)",
        )
        sel = None
        try:
            sel = self._window.create_file_dialog(
                open_dialog, allow_multiple=False, file_types=file_types)
        except Exception:
            try:  # fall back to an unfiltered dialog so Open CSV always works
                sel = self._window.create_file_dialog(
                    open_dialog, allow_multiple=False)
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

    def _store_transform(self, label: str, source: str, values,
                         method: str, unit: str = "",
                         params: dict | None = None) -> str:
        """Keep a full-resolution derived trace for later tools/export."""
        name = str(label or f"{source} {method}").strip()
        self._last_transforms[name] = {
            "source": str(source),
            "formula": "",
            "gain": 1.0,
            "offset": 0.0,
            "unit": unit or "",
            "method": method,
            "params": _json_safe(params or {}),
            "values": np.asarray(values, dtype=np.float64).copy(),
        }
        return name

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
        from data_quality import quality_report

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
        ycols = [c for c in cols if c != xcol]
        quality = quality_report(ld, xcol)
        import_report = _import_report_payload(ld, quality, path, xcol, ycols)
        return {
            "ok": True,
            "path": path,
            "name": os.path.basename(path),
            "columns": cols,
            "x_col": xcol,
            "y_cols": ycols,
            "units": {str(k): str(v) for k, v in (ld.units or {}).items()},
            "n_rows": int(ld.n_rows),
            "series": series,
            "quality": _quality_payload(quality),
            "import_report": import_report,
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
        stress_dir = _stress_examples_dir()
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
                "group": "Benchmark datasets",
                "tools": list(ds.get("tools", [])),
                "guide": _json_safe(EXAMPLE_GUIDES.get(f, {})),
            })
        try:
            stress = json.loads((stress_dir / "manifest.json").read_text(encoding="utf-8"))
            for ds in stress.get("datasets", []):
                f = str(ds.get("file", ""))
                if not f or not (stress_dir / f).exists():
                    continue
                items.append({
                    "file": f"stress/{f}",
                    "id": str(ds.get("id", "")),
                    "title": str(ds.get("title", f)),
                    "group": "Stress-test datasets",
                    "tools": list(ds.get("tools", [])),
                    "guide": _json_safe(ds.get("guide", {})),
                })
        except Exception:
            # Stress examples are useful, not required for the beginner tour.
            pass
        return {"ok": True, "dir": str(d), "stress_dir": str(stress_dir),
                "examples": items}

    def load_example(self, file: str):
        """Load one bundled example CSV by name (read-only; no path traversal)."""
        text = str(file or "")
        if text.startswith("stress/"):
            d = _stress_examples_dir()
            name = os.path.basename(text.split("/", 1)[1])
        else:
            d = _examples_dir()
            name = os.path.basename(text)
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
            {"id": "rlc_audit", "group": "Reconstruction", "name": "Reconstruction audit"},
            {"id": "calibration", "group": "Calibration", "name": "Forced-origin reference gain"},
            {"id": "cal_log", "group": "Calibration", "name": "Calibration log"},
            {"id": "export_data", "group": "Export", "name": "Export analyzed CSV"},
            {"id": "pipeline", "group": "Workflow", "name": "Analyze shot pipeline"},
            {"id": "selfcheck", "group": "Help", "name": "Toolbox self-check"},
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
                return {"ok": True, "text": rep.text(),
                        "one_line": rep.one_line(),
                        "status": rep.status,
                        "quality": _quality_payload(rep),
                        "read_only": True}

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
                label = self._store_transform(str(label), str(col), y, "lowpass",
                                              params={"cutoff_hz": cutoff})
                return {"ok": True, "label": label, "series": self._series(x, y),
                        "text": f"Applied deterministic low-pass filter at {cutoff:g} Hz to {col}.",
                        "read_only": True}

            if tool_id == "movmean":
                window = int(float(params.get("window", 101) or 101))
                y = movmean(self._column(str(col)), window)
                label = params.get("label") or f"{col} movmean {window}"
                label = self._store_transform(str(label), str(col), y, "movmean",
                                              params={"window": window})
                return {"ok": True, "label": label, "series": self._series(x, y),
                        "text": f"Applied moving average window={window} samples to {col}.",
                        "read_only": True}

            if tool_id == "gradient":
                y = gradient(self._column(str(col)), x)
                label = params.get("label") or f"d/dt {col}"
                label = self._store_transform(str(label), str(col), y, "gradient",
                                              unit=f"{self._loaded.units.get(str(col), '')}/s".strip("/"),
                                              params={})
                return {"ok": True, "label": label, "series": self._series(x, y),
                        "text": f"Computed derivative/dI-dt for {col} using NumPy gradient.",
                        "read_only": True}

            if tool_id == "integrate":
                y = integrate(self._column(str(col)), x)
                label = params.get("label") or f"integral {col}"
                label = self._store_transform(str(label), str(col), y, "integrate",
                                              params={})
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
                physical = _physical_rlc_params(params)
                rep = fit_rlc(x, y, sat_level=sat, label=str(col),
                              t_window=t_window, y_ref=yref,
                              ref_window=ref_window,
                              ref_label=str(ref_name or ""),
                              trusted_windows=_parse_windows(params.get("trusted_windows")),
                              **physical)
                return {"ok": rep.ok, "text": rep.text,
                        "overlay": rep.curve, "params": rep.params,
                        "read_only": True}

            if tool_id in ("rlc_audit", "reconstruction_audit"):
                from reconstruction_audit import audit_reconstruction
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
                physical = _physical_rlc_params(params)
                sens = _parse_optional_float(params.get("sensitivity_pct"))
                if sens is None:
                    sens = 0.10
                if sens > 1.0:
                    sens *= 0.01
                rep = audit_reconstruction(
                    x, y, label=str(col), sat_level=sat, y_ref=yref,
                    ref_label=str(ref_name or ""), t_window=t_window,
                    ref_window=ref_window,
                    trusted_windows=_parse_windows(params.get("trusted_windows")),
                    sensitivity_pct=max(0.0, min(float(sens), 0.50)),
                    run_sensitivity=_truthy(params.get("run_sensitivity", True)),
                    **physical)
                return {"ok": rep.ok, "text": rep.text,
                        "overlay": rep.overlay, "params": rep.params,
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
                ref_name = params.get("ref_channel") or params.get("reference")
                selected = []
                parts = []
                if _bool_param(params, "run_stats", True):
                    selected.append("statistics")
                    parts.append(self.column_stats(str(col)))
                if _bool_param(params, "run_anomaly", True):
                    selected.append("anomaly scan")
                    parts.append(self.run_tool("anomaly", {
                        "column": col,
                        "threshold_sigma": params.get("threshold_sigma", 6.0),
                        "t_start": params.get("t_start"),
                        "t_end": params.get("t_end"),
                    }))
                if _bool_param(params, "run_saturation", True):
                    selected.append("saturation estimate")
                    parts.append(self.run_tool("saturation", {
                        "column": col,
                        "sat_level": sat_level,
                        "ref_channel": ref_name,
                    }))
                if _bool_param(params, "run_rlc", True):
                    selected.append("RLC reconstruction")
                    parts.append(self.run_tool("rlc", {
                        "column": col,
                        "sat_level": sat_level,
                        "ref_channel": ref_name,
                        "t_start": params.get("t_start"),
                        "t_end": params.get("t_end"),
                        "ref_start": params.get("ref_start"),
                        "ref_end": ref_end,
                        "trusted_windows": params.get("trusted_windows", ""),
                        **_physical_rlc_params(params),
                    }))
                if _bool_param(params, "run_audit", False):
                    selected.append("reconstruction audit")
                    parts.append(self.run_tool("rlc_audit", {
                        "column": col,
                        "sat_level": sat_level,
                        "ref_channel": ref_name,
                        "t_start": params.get("t_start"),
                        "t_end": params.get("t_end"),
                        "ref_start": params.get("ref_start"),
                        "ref_end": ref_end,
                        "trusted_windows": params.get("trusted_windows", ""),
                        "sensitivity_pct": params.get("sensitivity_pct", 0.10),
                        "run_sensitivity": params.get("run_sensitivity", True),
                        **_physical_rlc_params(params),
                    }))
                if not parts:
                    return {"ok": False, "error": "choose at least one analysis",
                            "read_only": True}
                header = [
                    f"Analyze shot pipeline for {col}",
                    "Selected analyses: " + ", ".join(selected),
                    "Original CSV untouched; overlays are in-memory model/display layers.",
                ]
                physical = _physical_rlc_params(params)
                if any(physical[k] is not None for k in (
                        "resistance_ohm", "inductance_h",
                        "capacitance_f", "charging_voltage_v")):
                    header.append(
                        "Physical RLC hints: "
                        f"R={physical['resistance_ohm']} ohm, "
                        f"L={physical['inductance_h']} H, "
                        f"C={physical['capacitance_f']} F, "
                        f"V0={physical['charging_voltage_v']} V, "
                        f"soft-prior weight={physical['physical_prior_weight']}"
                    )
                text = "\n".join(header) + "\n\n" + "\n\n".join(
                    p.get("text") or json.dumps(_json_safe(p), indent=2)
                    for p in parts)
                overlay = next((p.get("overlay") for p in reversed(parts)
                                if p.get("ok") and p.get("overlay")), None)
                return {"ok": True, "text": text, "overlay": overlay,
                        "read_only": True}

            return {"ok": False, "error": f"unknown tool: {tool_id}"}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "read_only": True}

    def export_analyzed_csv(self, columns: list[str] | None = None,
                            path: str = ""):
        """Export selected raw/derived traces as a new CSV plus metadata.

        This is intentionally explicit and one-way: it never rewrites the
        original scope file. Derived traces come from the in-memory transform
        registry at full resolution, not from display decimation.
        """
        if self._loaded is None:
            return {"ok": False, "error": "load a CSV first", "read_only": True}
        try:
            out_path = str(path or "").strip()
            if not out_path:
                if self._window is None:
                    return {"ok": False, "error": "no save dialog available",
                            "read_only": True}
                import webview

                stem = Path(self._loaded.path).stem or "scope_data"
                sel = self._window.create_file_dialog(
                    webview.SAVE_DIALOG,
                    save_filename=f"{stem}_analyzed.csv",
                    file_types=("CSV files (*.csv)", "All files (*.*)"))
                if not sel:
                    return {"ok": False, "error": "cancelled", "read_only": True}
                out_path = sel[0] if isinstance(sel, (list, tuple)) else str(sel)
            if not out_path.lower().endswith(".csv"):
                out_path += ".csv"

            xcol = self._xcol()
            xname = xcol or "sample_index"
            x = self._x()
            requested = [str(c) for c in (columns or []) if str(c).strip()]
            if not requested:
                requested = [
                    str(c) for c in self._loaded.df.columns
                    if str(c) != str(xcol)
                ] + list(self._last_transforms.keys())

            import pandas as pd

            data = {}
            used = set()
            data[_unique_column_name(str(xname), used)] = x
            exported = []
            skipped = []
            for col in requested:
                if col == xcol:
                    continue
                try:
                    values = self._column(col)
                except KeyError:
                    skipped.append(col)
                    continue
                if len(values) != len(x):
                    skipped.append(col)
                    continue
                out_name = _unique_column_name(col, used)
                data[out_name] = values
                exported.append({"requested": col, "column": out_name})

            if not exported:
                return {"ok": False, "error": "no exportable columns selected",
                        "read_only": True}

            out = Path(out_path).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(data).to_csv(out, index=False)
            meta_path = out.with_suffix(out.suffix + ".meta.json")
            transforms = {
                name: {
                    k: v for k, v in spec.items()
                    if k != "values"
                }
                for name, spec in self._last_transforms.items()
                if not requested or name in requested
            }
            meta = {
                "created_by": "Scope Analyzer Lite",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_csv": self._loaded.path,
                "source_file_name": os.path.basename(self._loaded.path),
                "source_file_size_bytes": os.path.getsize(self._loaded.path),
                "read_only_source_csv": True,
                "output_csv": str(out),
                "columns_exported": exported,
                "columns_skipped": skipped,
                "x_column": str(xname),
                "units": {str(k): str(v) for k, v in (self._loaded.units or {}).items()},
                "transforms": _json_safe(transforms),
                "note": (
                    "Original CSV was not modified. This file is a derived "
                    "analysis export created after explicit user action."
                ),
            }
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
            return {
                "ok": True,
                "path": str(out),
                "metadata_path": str(meta_path),
                "n_rows": int(len(x)),
                "n_columns": int(len(data)),
                "columns_exported": exported,
                "columns_skipped": skipped,
                "text": (
                    f"Exported analyzed CSV:\n{out}\n\n"
                    f"Metadata sidecar:\n{meta_path}\n\n"
                    f"Rows: {len(x):,} · data columns: {len(exported)}\n"
                    "Original source CSV was not modified."
                ),
                "read_only": True,
            }
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

    def toolbox_self_check(self):
        """Run the bundled no-LLM toolbox benchmark from inside Lite.

        This is the in-app equivalent of scripts/benchmark_lite_toolbox.py. It
        verifies that the packaged examples are present, every deterministic
        tool can execute, and source CSV hashes remain unchanged.
        """
        d = _examples_dir()
        manifest = d / "manifest.json"
        if not manifest.exists():
            return {
                "ok": False,
                "read_only": True,
                "dir": str(d),
                "error": f"No benchmark manifest found at {manifest}",
                "text": (
                    "Toolbox self-check could not start because the bundled "
                    f"examples manifest was not found at:\n{manifest}\n\n"
                    "Try rebuilding the Lite app, or open the repo version and "
                    "run scripts/generate_lite_toolbox_examples.py."
                ),
            }
        try:
            from scripts.benchmark_lite_toolbox import run

            fails, results = run(d)
            passes = len(results) - fails
            lines = [
                "Scope Analyzer Lite in-app toolbox self-check",
                "============================================",
                f"Examples folder: {d}",
                f"Result: {passes} pass / {fails} fail",
                "",
                "Each case loads through the same Python bridge used by Lite,",
                "runs deterministic NumPy/SciPy tools, and verifies that the",
                "source CSV hash is unchanged.",
                "",
            ]
            for r in results:
                mark = "PASS" if r.get("ok") else "FAIL"
                lines.append(f"[{mark}] {r.get('file')} - {r.get('title')}")
                lines.append(f"       {r.get('detail')}")
            return {
                "ok": fails == 0,
                "read_only": True,
                "dir": str(d),
                "passes": passes,
                "fails": fails,
                "n": len(results),
                "results": _json_safe(results),
                "text": "\n".join(lines) + "\n",
            }
        except Exception as e:
            return {
                "ok": False,
                "read_only": True,
                "dir": str(d),
                "error": f"{type(e).__name__}: {e}",
                "text": (
                    "Toolbox self-check failed before completing.\n\n"
                    f"Examples folder: {d}\n"
                    f"Error: {type(e).__name__}: {e}\n\n"
                    "This does not modify any CSV; it only means the packaged "
                    "benchmark runner or one of its deterministic tools needs "
                    "attention."
                ),
            }

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
