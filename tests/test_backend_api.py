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
from pathlib import Path

import numpy as np
import pytest

# Make scope_web importable; backend_api adds the repo root itself on import.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scope_web"))

import backend_api  # noqa: E402
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
    assert st["read_only"] is True
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

# --- applied channel / tool bridge ----------------------------------------

def test_list_presets_exposes_bbcm_recipe():
    r = Api().list_presets()
    assert r["ok"] is True
    names = [p["name"] for p in r["presets"]]
    assert any("BBCM" in n for n in names)
    assert all({"name", "gain", "offset", "unit", "formula"}.issubset(p) for p in r["presets"])


def test_apply_channel_formula_is_derived_and_read_only(tek_csv):
    before = tek_csv.read_bytes()
    api = Api()
    api.load_csv(str(tek_csv))
    r = api.apply_channel("CH1", formula="(x-2.5)*10", gain=4, offset=1,
                          label="CH1 derived", unit="A")
    assert r["ok"] is True
    assert r["label"] == "CH1 derived"
    assert set(r["series"]) == {"x", "y"}
    assert len(r["series"]["x"]) == len(r["series"]["y"])
    assert r["read_only"] is True
    assert tek_csv.read_bytes() == before


def test_run_tool_anomaly_and_transform_are_soft_contracts(tek_csv):
    api = Api()
    api.load_csv(str(tek_csv))
    an = api.run_tool("anomaly", {"column": "CH2", "threshold_sigma": 6})
    assert an["ok"] is True and "text" in an
    lp = api.run_tool("lowpass", {"column": "CH2", "cutoff_hz": 10000})
    assert lp["ok"] is True
    assert "series" in lp and len(lp["series"]["x"]) == len(lp["series"]["y"])




def test_calibration_log_is_persistent_and_read_only(tmp_path, monkeypatch, tek_csv):
    log_path = tmp_path / "calibration_log.jsonl"
    monkeypatch.setenv("SCOPE_ANALYZER_CALIBRATION_LOG", str(log_path))
    before = tek_csv.read_bytes()
    api = Api()
    api.load_csv(str(tek_csv))

    saved = api.save_calibration_log({
        "kind": "display_formula",
        "source": "CH1",
        "preset_name": "BBCM test",
        "formula": "(x-2.5)*750",
        "gain": 4,
        "offset": 0,
        "unit": "A",
    })

    assert saved["ok"] is True
    assert saved["read_only"] is True
    assert log_path.exists()
    listed = api.list_calibration_log()
    assert listed["ok"] is True
    assert listed["entries"][0]["preset_name"] == "BBCM test"
    assert listed["entries"][0]["shot"]["file_name"] == "T0000.CSV"
    assert tek_csv.read_bytes() == before


def test_list_tools_has_release_menu_groups():
    r = Api().list_tools()
    assert r["ok"] is True
    ids = {t["id"] for t in r["tools"]}
    assert {"stats", "formula", "anomaly", "saturation", "rlc", "calibration", "fft"}.issubset(ids)



def test_run_tool_alias_analyze_executes_pipeline(tek_csv):
    api = Api()
    api.load_csv(str(tek_csv))
    r = api.run_tool("analyze", {"column": "CH2"})
    assert r["ok"] is True
    assert "peak" in r["text"] or "CH2" in r["text"]
    assert r["read_only"] is True


def test_tools_can_use_display_derived_trace(tek_csv):
    api = Api()
    api.load_csv(str(tek_csv))
    derived = api.apply_channel("CH1", formula="(x-2.5)*10", gain=4, offset=1,
                                label="CH1 calibrated display", unit="A")
    assert derived["ok"] is True
    st = api.run_tool("stats", {"column": "CH1 calibrated display"})
    assert st["ok"] is True
    assert st["column"] == "CH1 calibrated display"
    lp = api.run_tool("lowpass", {"column": "CH1 calibrated display", "cutoff_hz": 10000})
    assert lp["ok"] is True
    assert lp["read_only"] is True


def test_list_and_load_examples(tmp_path, monkeypatch):
    """Examples menu bridge: manifest lists datasets and load_example loads one
    read-only, with a path-traversal guard."""
    from scripts.generate_lite_toolbox_examples import make_examples
    out = tmp_path / "tool_benchmarks"
    make_examples(out)
    monkeypatch.setenv("SCOPE_ANALYZER_EXAMPLES", str(out))
    api = Api()
    lst = api.list_examples()
    assert lst["ok"] is True
    assert len(lst["examples"]) == 15
    assert all({"file", "id", "title", "tools"}.issubset(e) for e in lst["examples"])
    first = lst["examples"][0]["file"]
    r = api.load_example(first)
    assert r["ok"] is True and r["read_only"] is True and r["series"] and r["x_col"]
    bad = api.load_example("../../etc/passwd")
    assert bad["ok"] is False


def test_resource_root_finds_macos_bundle_resources(tmp_path, monkeypatch):
    """Frozen macOS apps keep data files under Contents/Resources, while the
    Python runtime may report a different _MEIPASS folder. The examples menu
    depends on this resolving to Resources."""
    contents = tmp_path / "ScopeAnalyzerLite.app" / "Contents"
    resources = contents / "Resources"
    frameworks = contents / "Frameworks"
    macos = contents / "MacOS"
    (resources / "examples").mkdir(parents=True)
    frameworks.mkdir(parents=True)
    macos.mkdir(parents=True)
    exe = macos / "ScopeAnalyzerLite"
    exe.write_text("")

    monkeypatch.setattr(backend_api.sys, "frozen", True, raising=False)
    monkeypatch.setattr(backend_api.sys, "executable", str(exe), raising=False)
    monkeypatch.setattr(backend_api.sys, "_MEIPASS", str(frameworks), raising=False)

    assert Path(backend_api._resource_root()) == resources
