"""
Contract tests for the web-frontend bridge (scope_web/backend_api.py).

The bridge is the seam the HTML/JS UI is built on: JS calls
``window.pywebview.api.<method>`` and depends on the SHAPE of the dict that
comes back. These tests freeze that shape so the contract cannot drift
silently once UI is wired onto it. They also pin two properties the science
relies on: source data is never mutated, and min-max decimation preserves the
true peak/trough (a naive every-Nth decimation would hide the spikes that
matter most in a pulsed-power capture).

Headless: no pywebview, no Qt. The Api is importable and exercised directly.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Make scope_web importable; backend_api adds the repo root itself on import.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scope_web"))

from backend_api import Api, PLOT_POINTS  # noqa: E402


# --- fixtures --------------------------------------------------------------

@pytest.fixture
def big_spike_csv(tmp_path):
    """A large single-channel capture with ONE sharp spike, so the decimation
    peak-preservation invariant has something a naive decimator would drop."""
    n = 20_000
    t = np.linspace(0.0, 0.05, n)          # 0..50 ms
    y = np.zeros(n)
    y[12_345] = 6600.0                     # a lone 6.6 kA spike between samples
    y[12_346] = -200.0                     # and a lone trough next to it
    lines = ["TIME,CH1"]
    lines += [f"{t[i]:.9e},{y[i]:.6e}" for i in range(n)]
    p = tmp_path / "T0001.CSV"
    p.write_text("\n".join(lines) + "\n")
    return p, float(y.max()), float(y.min())


# --- load_csv return-shape contract ---------------------------------------

LOAD_KEYS = {"ok", "path", "name", "columns", "x_col", "y_cols",
             "units", "n_rows", "series"}


def test_load_csv_contract_shape(tek_csv):
    api = Api()
    r = api.load_csv(str(tek_csv))
    assert r["ok"] is True
    # every documented key present, no surprises removed
    assert LOAD_KEYS.issubset(r.keys())
    assert r["name"] == "T0000.CSV"
    assert r["x_col"] == "TIME"                       # time column detected
    assert r["columns"][0] == "TIME"
    assert set(r["y_cols"]) == {"CH1", "CH2"}
    assert r["x_col"] not in r["y_cols"]              # time is never a series
    assert r["n_rows"] == 200
    # units mapped from the Tektronix 'Vertical Units' preamble
    assert r["units"].get("CH1") == "V"
    assert r["units"].get("CH2") == "A"


def test_load_csv_series_are_paired_and_decimated(tek_csv):
    api = Api()
    r = api.load_csv(str(tek_csv))
    assert set(r["series"].keys()) == set(r["y_cols"])
    for name, s in r["series"].items():
        assert set(s.keys()) == {"x", "y"}
        assert len(s["x"]) == len(s["y"])             # paired
        assert len(s["y"]) <= PLOT_POINTS + 4         # decimation cap
        # JSON-serialisable plain floats, not numpy types
        assert all(isinstance(v, float) for v in s["y"][:5])


def test_load_csv_bad_path_is_soft_error():
    api = Api()
    r = api.load_csv("/no/such/file.CSV")
    assert r["ok"] is False
    assert "error" in r                                # honest failure, no raise


# --- decimation preserves the science -------------------------------------

def test_decimation_preserves_peak_and_trough(big_spike_csv):
    path, true_max, true_min = big_spike_csv
    api = Api()
    r = api.load_csv(str(path))
    ys = r["series"]["CH1"]["y"]
    assert len(ys) <= PLOT_POINTS + 4                  # actually decimated
    # the lone spike and trough survive the envelope decimation exactly
    assert max(ys) == pytest.approx(true_max)
    assert min(ys) == pytest.approx(true_min)


def test_load_does_not_mutate_source(tek_csv):
    before = tek_csv.read_bytes()
    Api().load_csv(str(tek_csv))
    assert tek_csv.read_bytes() == before              # cardinal rule


# --- column_stats contract -------------------------------------------------

STATS_KEYS = {"ok", "column", "n", "n_finite", "min", "max", "mean", "std",
              "median", "p5", "p95", "rms", "window"}


def test_column_stats_contract_and_sanity(tek_csv):
    api = Api()
    api.load_csv(str(tek_csv))
    st = api.column_stats("CH2")                        # clean 0..1000 ramp
    assert STATS_KEYS.issubset(st.keys())
    assert st["ok"] is True
    assert st["n"] == 200 and st["n_finite"] == 200
    assert st["min"] == pytest.approx(0.0, abs=1e-3)
    assert st["max"] == pytest.approx(1000.0, rel=1e-4)
    assert st["min"] <= st["p5"] <= st["median"] <= st["p95"] <= st["max"]
    assert st["rms"] >= 0.0
    # full-range window equals the file's time span
    assert st["window"][0] == pytest.approx(-1e-3)
    assert st["window"][1] == pytest.approx(1e-3)


def test_column_stats_window_subsets(tek_csv):
    api = Api()
    api.load_csv(str(tek_csv))
    full = api.column_stats("CH2")
    half = api.column_stats("CH2", t_start=0.0)         # only t >= 0
    assert half["ok"] is True
    assert half["n_finite"] < full["n_finite"]          # window narrowed it
    assert half["window"][0] >= 0.0
    assert half["min"] >= full["min"] - 1e-6            # upper half of the ramp


def test_column_stats_unknown_column_is_soft_error(tek_csv):
    api = Api()
    api.load_csv(str(tek_csv))
    st = api.column_stats("NOPE")
    assert st["ok"] is False and "error" in st
