"""
signal_tools.py - Safe expression transforms for oscilloscope channels.

The goal is to let Scope Studio apply MATLAB-like monitor conversions
without hard-coding every possible current sensor. Formulas are evaluated
against a restricted set of NumPy-backed helpers and never get Python
builtins or attribute access.
"""
from __future__ import annotations

import ast
import math

import numpy as np

try:
    from numba import njit
except Exception:  # numba is optional; NumPy/SciPy paths still work.
    njit = None


class FormulaError(ValueError):
    """Raised when a channel formula is invalid or unsafe."""


def baseline(y: np.ndarray, axis: np.ndarray, end: float,
             start: float | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    if start is None:
        mask = axis <= end
    else:
        mask = (axis >= start) & (axis <= end)
    if not np.any(mask):
        return y
    return y - float(np.nanmean(y[mask]))


def demean(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return y - float(np.nanmean(y))


def movmean(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return y
    window = min(max(int(window), 1), y.size)
    if window <= 1:
        return y
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y, kernel, mode="same")


def gradient(y: np.ndarray, axis: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    if y.size < 2:
        return np.zeros_like(y, dtype=np.float64)
    return np.gradient(y, axis)


def integrate(y: np.ndarray, axis: np.ndarray,
              negate: bool = False) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    if y.size == 0:
        return y
    out = np.zeros_like(y, dtype=np.float64)
    if y.size > 1:
        seg = 0.5 * (y[1:] + y[:-1]) * np.diff(axis)
        out[1:] = np.cumsum(seg)
    return -out if negate else out


def _butter_lowpass(y: np.ndarray, dt: float,
                    cutoff_hz: float) -> np.ndarray | None:
    try:
        from scipy.signal import butter, filtfilt
    except Exception:
        return None
    fs = 1.0 / dt
    nyq = 0.5 * fs
    wn = min(max(cutoff_hz / nyq, 1e-6), 0.99)
    b, a = butter(2, wn, btype="low")
    return filtfilt(b, a, y)


def _rc_lowpass_core(y: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(y, dtype=np.float64)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = out[i - 1] + alpha * (y[i] - out[i - 1])
    rev = np.empty_like(out, dtype=np.float64)
    rev[-1] = out[-1]
    for i in range(len(out) - 2, -1, -1):
        rev[i] = rev[i + 1] + alpha * (out[i] - rev[i + 1])
    return rev


if njit is not None:
    _rc_lowpass_core = njit(cache=True)(_rc_lowpass_core)


def _rc_lowpass(y: np.ndarray, dt: float, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0 or dt <= 0:
        return y
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    return _rc_lowpass_core(y, alpha)


def lowpass(y: np.ndarray, axis: np.ndarray, cutoff_hz: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    if y.size < 3:
        return y
    diffs = np.diff(axis)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return y
    dt = float(np.median(diffs))
    filtered = _butter_lowpass(y, dt, float(cutoff_hz))
    if filtered is not None:
        return np.asarray(filtered, dtype=np.float64)
    return _rc_lowpass(y, dt, float(cutoff_hz))


def amplitude_spectrum(y: np.ndarray, dt: float):
    """Single-sided amplitude spectrum of ``y`` sampled every ``dt`` seconds.

    Hann-windowed with coherent-gain (window-sum) normalization, so a pure
    tone of amplitude A that is bin-centered reads ~A at its frequency.
    Returns ``(freq_hz, amp)``; empty arrays if there is too little data.

    This is the pure, testable core of the Detail+FFT ringing view (used by
    surface3d), so the spectrum math is exercised by the headless test
    suite rather than living only inside a Qt method.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 2 or dt <= 0:
        return np.zeros(0), np.zeros(0)
    win = np.hanning(n)
    amp = np.abs(np.fft.rfft(y * win)) * 2.0 / max(float(win.sum()), 1.0)
    freq = np.fft.rfftfreq(n, d=dt)
    return freq, amp


def dominant_frequency(y: np.ndarray, dt: float, f_min: float = 0.0) -> float:
    """Frequency (Hz) of the largest spectral peak above ``f_min``."""
    freq, amp = amplitude_spectrum(y, dt)
    if freq.size == 0:
        return 0.0
    mask = freq > f_min
    if not np.any(mask):
        return 0.0
    idx = np.flatnonzero(mask)
    k = int(idx[int(np.argmax(amp[idx]))])
    return float(freq[k])


SAFE_FUNCS = {
    "abs": np.abs,
    "baseline": baseline,
    "clip": np.clip,
    "cos": np.cos,
    "demean": demean,
    "exp": np.exp,
    "gradient": gradient,
    "integrate": integrate,
    "log": np.log,
    "log10": np.log10,
    "lowpass": lowpass,
    "max": np.maximum,
    "min": np.minimum,
    "movmean": movmean,
    "sign": np.sign,
    "sin": np.sin,
    "sqrt": np.sqrt,
    "tan": np.tan,
    "where": np.where,
}

SAFE_CONSTS = {
    "e": math.e,
    "pi": math.pi,
}


def _samples_from_dt(seconds: float, dt: float,
                     minimum: int = 1) -> int:
    try:
        seconds = abs(float(seconds))
        minimum = max(int(minimum), 1)
    except (TypeError, ValueError):
        return 1
    if not math.isfinite(seconds) or dt <= 0 or not math.isfinite(dt):
        return minimum
    return max(int(round(seconds / dt)), minimum)

ALLOWED_NODES = (
    ast.Add,
    ast.And,
    ast.BinOp,
    ast.BoolOp,
    ast.Call,
    ast.Compare,
    ast.Constant,
    ast.Div,
    ast.Eq,
    ast.Expression,
    ast.Gt,
    ast.GtE,
    ast.IfExp,
    ast.keyword,
    ast.Lt,
    ast.LtE,
    ast.Load,
    ast.Mod,
    ast.Mult,
    ast.Name,
    ast.NotEq,
    ast.Or,
    ast.Pow,
    ast.Sub,
    ast.Tuple,
    ast.UAdd,
    ast.UnaryOp,
    ast.USub,
)


def _validate_formula(formula: str, allowed_names: set[str]) -> ast.Expression:
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"Formula syntax error: {exc.msg}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise FormulaError(
                f"Unsupported syntax in formula: {type(node).__name__}"
            )
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise FormulaError(f"Unknown name in formula: {node.id}")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise FormulaError("Only direct helper-function calls are allowed.")
    return tree


def evaluate_formula(formula: str, x: np.ndarray, t_s: np.ndarray,
                     t_ms: np.ndarray | None = None,
                     columns: dict | None = None) -> np.ndarray:
    formula = (formula or "x").strip() or "x"
    x = np.asarray(x, dtype=np.float64)
    t_s = np.asarray(t_s, dtype=np.float64)
    if t_ms is None:
        t_ms = t_s * 1000.0
    else:
        t_ms = np.asarray(t_ms, dtype=np.float64)
    dt = float(np.median(np.diff(t_s))) if t_s.size > 1 else 0.0

    env = {
        "x": x,
        "t": t_s,
        "t_ms": t_ms,
        "dt": dt,
        "dt_ms": dt * 1000.0,
        "samples": lambda seconds, minimum=1: _samples_from_dt(
            seconds, dt, minimum
        ),
        **SAFE_FUNCS,
        **SAFE_CONSTS,
    }
    # Expose the document's other columns so formulas can do column-to-column
    # arithmetic, e.g. ``CH1 - CH2`` or ``col("CH1 Peak Detect") / CH2``.
    if columns:
        colmap = {str(nm): np.asarray(arr, dtype=np.float64)
                  for nm, arr in columns.items()}

        def _col(name):
            key = str(name)
            if key not in colmap:
                raise FormulaError(f"Unknown column: {name}")
            return colmap[key]

        env["col"] = _col
        for nm, arr in colmap.items():
            if nm.isidentifier() and nm not in env:
                env[nm] = arr
    tree = _validate_formula(formula, set(env))
    try:
        out = eval(compile(tree, "<formula>", "eval"),
                   {"__builtins__": {}}, env)
    except Exception as exc:
        raise FormulaError(str(exc)) from exc

    out = np.asarray(out, dtype=np.float64)
    if out.ndim == 0:
        raise FormulaError("Formula must return an array, not a scalar.")
    if len(out) != len(x):
        raise FormulaError(
            f"Formula returned {len(out)} samples, expected {len(x)}."
        )
    return out


def combine_columns(a: np.ndarray, b: np.ndarray, op: str,
                    eps: float = 1e-9) -> np.ndarray:
    """Safe element-wise arithmetic between two columns A (op) B.

    Operators: ``+ - * /`` (aliases add/sub/mul/div). Refuses, with a clear
    FormulaError, when the inputs are mismatched, when a divisor is zero or
    near-zero (within ``eps``), or when the result is not finite (NaN/Inf).
    This is the guarded path the UI uses; the source CSV is never modified.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise FormulaError(
            f"Columns have different lengths ({a.size} vs {b.size}).")
    key = str(op).strip().lower()
    if key in ("+", "add", "plus"):
        out = a + b
    elif key in ("-", "sub", "subtract", "minus"):
        out = a - b
    elif key in ("*", "x", "mul", "multiply", "times"):
        out = a * b
    elif key in ("/", "div", "divide", "over"):
        bad = ~np.isfinite(b) | (np.abs(b) < eps)
        n_bad = int(np.count_nonzero(bad))
        if n_bad:
            raise FormulaError(
                f"Division by zero / near-zero at {n_bad} sample(s); refused.")
        out = a / b
    else:
        raise FormulaError(f"Unknown operator {op!r} (use + - * /).")
    n_nonfinite = int(np.count_nonzero(~np.isfinite(out)))
    if n_nonfinite:
        raise FormulaError(
            f"Result has {n_nonfinite} NaN/Inf value(s); refused.")
    return out
