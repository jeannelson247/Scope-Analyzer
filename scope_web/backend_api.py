"""backend_api.py — the Python<->JS bridge for the web-frontend Scope Studio.

This is the seam between the Claude-Design UI (HTML/JS) and the REAL Scope
Studio backend. Every method here is callable from JS as
`window.pywebview.api.<method>(...)` and returns plain JSON-able dicts.

Design rule (mirrors the native app): the browser is only a view. All data
loading and computation stays in Python, reusing the existing, tested modules
(csv_loader, signal_tools, detect_anomalies, calibration, ai_assistant). The
UI never sees a raw 125k-row file — it gets decimated series ready to draw.

Importing this module does NOT require pywebview, so the backend half can be
unit-tested headlessly (see `selftest`).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csv_loader import load_csv, minmax_decimate  # noqa: E402

PLOT_POINTS = 4000          # decimation target — smooth on screen, light on the bridge


class Api:
    """Methods exposed to JavaScript. Keep returns JSON-serializable."""

    def __init__(self):
        self._window = None
        self._loaded = None          # last LoadedData

    def set_window(self, window):
        self._window = window

    # -- file open ---------------------------------------------------------
    def pick_csv(self):
        """Open the native file dialog (via pywebview) and load the choice."""
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

    # -- data --------------------------------------------------------------
    def load_csv(self, path: str):
        """Load a CSV via the real loader and return decimated series + meta."""
        try:
            ld = load_csv(path)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self._loaded = ld
        df = ld.df
        cols = [str(c) for c in df.columns]
        xcol = next((c for c in cols if "time" in c.lower()), None)
        x = (df[xcol].to_numpy() if xcol else
             np.arange(len(df), dtype=float))
        series = {}
        for c in cols:
            if c == xcol:
                continue
            y = df[c].to_numpy().astype(float)
            xs, ys = minmax_decimate(x, y, PLOT_POINTS)
            series[c] = {"x": [float(v) for v in xs],
                         "y": [float(v) for v in ys]}
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
        }

    # -- stats (reuses numpy; deterministic, no LLM) -----------------------
    def column_stats(self, column: str, t_start=None, t_end=None):
        """Deterministic summary stats for one column.

        Optional t_start/t_end (in the file's time units) restrict the stats to
        a window — e.g. the currently visible plot range — using the detected
        time column. The returned ``window`` records the actual [min, max] time
        the stats were computed over (None if there is no time column), so a
        figure caption can state exactly what range the numbers describe.

        Contract (keys are stable; UI may rely on them):
          ok, column, n, n_finite, min, max, mean, std, median, p5, p95, rms, window
        """
        if self._loaded is None or column not in self._loaded.df.columns:
            return {"ok": False, "error": "no such column / no data"}
        df = self._loaded.df
        y = df[column].to_numpy().astype(float)

        xcol = next((c for c in df.columns if "time" in str(c).lower()), None)
        window = None
        if xcol is not None:
            x = df[xcol].to_numpy().astype(float)
            if t_start is not None or t_end is not None:
                mask = np.ones(x.shape, dtype=bool)
                if t_start is not None:
                    mask &= x >= float(t_start)
                if t_end is not None:
                    mask &= x <= float(t_end)
                y = y[mask]
                x = x[mask]
            if x.size:
                window = [float(np.nanmin(x)), float(np.nanmax(x))]

        finite = y[np.isfinite(y)]
        if finite.size == 0:
            return {"ok": False, "error": "no finite samples in range",
                    "column": column, "window": window}
        return {"ok": True, "column": column,
                "n": int(y.size), "n_finite": int(finite.size),
                "min": float(np.min(finite)), "max": float(np.max(finite)),
                "mean": float(np.mean(finite)), "std": float(np.std(finite)),
                "median": float(np.median(finite)),
                "p5": float(np.percentile(finite, 5)),
                "p95": float(np.percentile(finite, 95)),
                "rms": float(np.sqrt(np.mean(np.square(finite)))),
                "window": window}

    # -- placeholders to wire next (kept explicit, not silently missing) ---
    def list_models(self):
        return {"ok": True, "models": [], "note": "wire to ai_assistant next"}

    def chat(self, prompt: str, backend: str = "mlx", model: str = ""):
        return {"ok": False, "error": "chat bridge not wired yet"}


def selftest() -> int:
    """Headless check of the data path on a real CSV (no pywebview, no GUI)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    candidates = [
        os.path.join(root, "Claidev2", "uploads", "T0026.CSV"),
    ]
    # fall back to any CSV under examples/
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
    print(f"loaded: {r.get('name')}  rows={r.get('n_rows')}  "
          f"cols={r.get('columns')}  x={r.get('x_col')}  "
          f"series={list(r.get('series', {}).keys())}")
    if fails:
        print("BACKEND-API SELFTEST: FAIL")
        for f in fails:
            print("  [FAIL]", f)
        return 1
    print("BACKEND-API SELFTEST: PASS (real CSV -> decimated series + stats)")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
