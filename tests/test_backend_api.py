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

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    assert r["quality"]["status"] == "ok"
    assert r["import_report"]["read_only"] is True
    assert r["import_report"]["delimiter_name"] == "comma"
    assert r["import_report"]["skiprows"] == 4
    assert "Read-only source CSV: yes" in r["import_report"]["text"]


def test_load_csv_import_report_for_semicolon_scope_export(tmp_path):
    p = tmp_path / "rigol_semicolon.csv"
    p.write_text(
        "\n".join([
            "Model;RIGOL-DS",
            "Vertical Units;V;A",
            "Source;CHAN1;CHAN2",
            "Time;CHAN1;CHAN2",
            "0.000000e+00;1.0;10.0",
            "1.000000e-06;1.5;20.0",
            "2.000000e-06;2.0;30.0",
        ]) + "\n",
        encoding="utf-8",
    )

    r = Api().load_csv(str(p))

    assert r["ok"] is True
    assert r["x_col"] == "Time"
    assert r["y_cols"] == ["CHAN1", "CHAN2"]
    assert r["units"] == {"CHAN1": "V", "CHAN2": "A"}
    assert r["quality"]["status"] == "ok"
    assert r["import_report"]["delimiter_name"] == "semicolon"
    assert r["import_report"]["scope_model"] == "RIGOL-DS"
    assert "Rows x columns: 3 x 3" in r["import_report"]["text"]


def test_load_csv_import_report_for_tab_scope_export(tmp_path):
    p = tmp_path / "keysight_tab.tsv"
    p.write_text(
        "\n".join([
            "Model\tKEYSIGHT",
            "Time\tVoltage",
            "0.000000e+00\t0.0",
            "1.000000e-06\t0.2",
            "2.000000e-06\t0.4",
        ]) + "\n",
        encoding="utf-8",
    )

    r = Api().load_csv(str(p))

    assert r["ok"] is True
    assert r["x_col"] == "Time"
    assert r["y_cols"] == ["Voltage"]
    assert r["quality"]["status"] == "ok"
    assert r["import_report"]["delimiter_name"] == "tab"
    assert r["import_report"]["scope_model"] == "KEYSIGHT"
    assert "Read-only source CSV: yes" in r["import_report"]["text"]


def test_pick_csv_dialog_accepts_csv_txt_and_tsv_scope_exports(tmp_path):
    p = tmp_path / "scope_export.tsv"
    p.write_text(
        "Time\tCH1\n0.0\t1.0\n1.0e-6\t2.0\n2.0e-6\t3.0\n",
        encoding="utf-8",
    )
    seen = {}

    class FakeWindow:
        def create_file_dialog(self, _dialog_type, **kwargs):
            seen.update(kwargs)
            return [str(p)]

    api = Api()
    api.set_window(FakeWindow())
    r = api.pick_csv()

    assert r["ok"] is True
    assert r["name"] == "scope_export.tsv"
    assert r["import_report"]["delimiter_name"] == "tab"
    filters = " ".join(seen["file_types"])
    assert "*.csv" in filters
    assert "*.txt" in filters
    assert "*.tsv" in filters


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
    assert {"stats", "formula", "anomaly", "saturation", "rlc", "rlc_audit",
            "calibration", "fft", "export_data", "help", "selfcheck"}.issubset(ids)


def test_list_examples_regenerates_missing_pack(tmp_path, monkeypatch):
    out_dir = tmp_path / "generated_examples"
    stress_dir = tmp_path / "generated_stress"
    monkeypatch.setenv("SCOPE_ANALYZER_EXAMPLES", str(out_dir))
    monkeypatch.setenv("SCOPE_ANALYZER_STRESS_EXAMPLES", str(stress_dir))

    r = Api().list_examples()

    assert r["ok"] is True
    assert Path(r["dir"]) == out_dir
    assert (out_dir / "manifest.json").exists()
    assert (stress_dir / "manifest.json").exists()
    assert len([e for e in r["examples"] if e["group"] == "Benchmark datasets"]) == 15
    assert len([e for e in r["examples"] if e["group"] == "Stress-test datasets"]) == 12
    assert (out_dir / "02_bbcm_clipped_6ka.csv").exists()
    assert (stress_dir / "stress_06_censored_multiwindow_6ka.csv").exists()



def test_run_tool_alias_analyze_executes_pipeline(tek_csv):
    api = Api()
    api.load_csv(str(tek_csv))
    r = api.run_tool("analyze", {"column": "CH2"})
    assert r["ok"] is True
    assert "peak" in r["text"] or "CH2" in r["text"]
    assert r["read_only"] is True


def test_rlc_accepts_physical_rlc_hints(tmp_path):
    from scripts.generate_lite_toolbox_examples import make_examples

    out = tmp_path / "tool_benchmarks"
    make_examples(out)
    api = Api()
    api.load_csv(str(out / "02_bbcm_clipped_6ka.csv"))

    r = api.run_tool("rlc", {
        "column": "BBCM_A",
        "sat_level": 6000,
        "ref_channel": "Pearson_A",
        "ref_start": 0.0,
        "ref_end": 0.010,
        "trusted_windows": "0:0.005, 0.040:0.150",
        "resistance_ohm": 0.055,
        "inductance_h": 166e-6,
        "capacitance_f": 2.24,
        "charging_voltage_v": 450,
        "physical_prior_weight": 0.15,
    })

    assert r["ok"] is True
    assert "physical RLC inputs" in r["text"]
    assert "expected tau_rise = L/R" in r["text"]
    assert "expected initial dI/dt = |V0|/L" in r["text"]
    assert r["params"]["physical"]["resistance_ohm"] == pytest.approx(0.055)
    assert r["params"]["physical"]["inductance_h"] == pytest.approx(166e-6)
    assert r["params"]["physical"]["capacitance_f"] == pytest.approx(2.24)
    assert r["params"]["physical"]["charging_voltage_v"] == pytest.approx(450)
    assert r["params"]["physical"]["expected_initial_slope_a_per_s"] == pytest.approx(450 / 166e-6)
    assert r["overlay"] and r["overlay"]["t"]


def test_reconstruction_audit_scores_methods_and_physics(tmp_path):
    from scripts.generate_lite_toolbox_examples import make_examples

    out = tmp_path / "tool_benchmarks"
    make_examples(out)
    api = Api()
    api.load_csv(str(out / "02_bbcm_clipped_6ka.csv"))

    r = api.run_tool("rlc_audit", {
        "column": "BBCM_A",
        "sat_level": 6000,
        "ref_channel": "Pearson_A",
        "ref_start": 0.0,
        "ref_end": 0.010,
        "trusted_windows": "0:0.005, 0.040:0.150",
        "resistance_ohm": 0.055,
        "inductance_h": 166e-6,
        "capacitance_f": 2.24,
        "charging_voltage_v": 450,
        "physical_prior_weight": 0.15,
        "sensitivity_pct": 10,
    })

    assert r["ok"] is True
    assert "Reconstruction audit" in r["text"]
    assert "Verdict:" in r["text"]
    assert "Peak estimates" in r["text"]
    assert "Physical consistency" in r["text"]
    assert "Sensitivity sweep" in r["text"]
    assert r["params"]["estimates"]["censored RLC"] == pytest.approx(
        r["params"]["rlc"]["peak"])
    assert r["params"]["sensitivity"]["ran"] is True
    assert r["overlay"] and r["overlay"]["t"]
    assert r["read_only"] is True


def test_analyze_pipeline_can_select_steps_and_pass_physical_rlc(tmp_path):
    from scripts.generate_lite_toolbox_examples import make_examples

    out = tmp_path / "tool_benchmarks"
    make_examples(out)
    api = Api()
    api.load_csv(str(out / "02_bbcm_clipped_6ka.csv"))

    r = api.run_tool("pipeline", {
        "column": "BBCM_A",
        "run_stats": False,
        "run_anomaly": False,
        "run_saturation": False,
        "run_rlc": True,
        "sat_level": 6000,
        "ref_channel": "Pearson_A",
        "ref_start": 0.0,
        "ref_end": 0.010,
        "trusted_windows": "0:0.005, 0.040:0.150",
        "resistance_ohm": 0.055,
        "inductance_h": 166e-6,
        "capacitance_f": 2.24,
        "charging_voltage_v": 450,
        "physical_prior_weight": 0.15,
    })

    assert r["ok"] is True
    assert "Selected analyses: RLC reconstruction" in r["text"]
    assert "Physical RLC hints" in r["text"]
    assert "V0=450.0 V" in r["text"]
    assert "Anomaly scan" not in r["text"]
    assert r["overlay"] and r["read_only"] is True


def test_analyze_pipeline_can_include_reconstruction_audit(tmp_path):
    from scripts.generate_lite_toolbox_examples import make_examples

    out = tmp_path / "tool_benchmarks"
    make_examples(out)
    api = Api()
    api.load_csv(str(out / "02_bbcm_clipped_6ka.csv"))

    r = api.run_tool("pipeline", {
        "column": "BBCM_A",
        "run_stats": False,
        "run_anomaly": False,
        "run_saturation": False,
        "run_rlc": False,
        "run_audit": True,
        "sat_level": 6000,
        "ref_channel": "Pearson_A",
        "ref_start": 0.0,
        "ref_end": 0.010,
        "trusted_windows": "0:0.005, 0.040:0.150",
        "resistance_ohm": 0.055,
        "inductance_h": 166e-6,
        "capacitance_f": 2.24,
        "charging_voltage_v": 450,
        "physical_prior_weight": 0.15,
        "sensitivity_pct": 10,
    })

    assert r["ok"] is True
    assert "Selected analyses: reconstruction audit" in r["text"]
    assert "Reconstruction audit" in r["text"]
    assert "Verdict:" in r["text"]
    assert r["overlay"] and r["read_only"] is True


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


def test_run_tool_transform_is_reusable_and_exportable(tmp_path, tek_csv):
    before = tek_csv.read_bytes()
    api = Api()
    api.load_csv(str(tek_csv))

    lp = api.run_tool("lowpass", {"column": "CH2", "cutoff_hz": 10000})
    assert lp["ok"] is True
    label = lp["label"]

    # Tool-created derived traces should be full-resolution backend state, not
    # only decimated display pixels.
    st = api.run_tool("stats", {"column": label})
    assert st["ok"] is True
    assert st["n"] == 200

    out = tmp_path / "analyzed_export.csv"
    saved = api.export_analyzed_csv(["CH2", label], str(out))
    assert saved["ok"] is True
    assert saved["read_only"] is True
    assert saved["n_rows"] == 200
    assert out.exists()
    meta = Path(saved["metadata_path"])
    assert meta.exists()

    exported = pd.read_csv(out)
    assert list(exported.columns) == ["TIME", "CH2", label]
    assert len(exported) == 200
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    assert metadata["read_only_source_csv"] is True
    assert metadata["transforms"][label]["method"] == "lowpass"
    assert metadata["transforms"][label]["params"]["cutoff_hz"] == 10000
    assert tek_csv.read_bytes() == before


def test_list_and_load_examples(tmp_path, monkeypatch):
    """Examples menu bridge: manifest lists datasets and load_example loads one
    read-only, with a path-traversal guard."""
    from scripts.generate_lite_toolbox_examples import make_examples
    from scripts.generate_lite_stress_examples import make_stress_examples
    out = tmp_path / "tool_benchmarks"
    stress = tmp_path / "tool_stress"
    make_examples(out)
    make_stress_examples(stress)
    monkeypatch.setenv("SCOPE_ANALYZER_EXAMPLES", str(out))
    monkeypatch.setenv("SCOPE_ANALYZER_STRESS_EXAMPLES", str(stress))
    api = Api()
    lst = api.list_examples()
    assert lst["ok"] is True
    assert len(lst["examples"]) == 27
    assert all({"file", "id", "title", "group", "tools", "guide"}.issubset(e) for e in lst["examples"])
    assert all(e["guide"].get("tool") for e in lst["examples"])
    assert all(e["guide"].get("column") for e in lst["examples"])
    first = lst["examples"][0]["file"]
    r = api.load_example(first)
    assert r["ok"] is True and r["read_only"] is True and r["series"] and r["x_col"]
    stress_file = next(e["file"] for e in lst["examples"] if e["group"] == "Stress-test datasets")
    rs = api.load_example(stress_file)
    assert rs["ok"] is True and rs["read_only"] is True and rs["series"] and rs["x_col"]
    bad = api.load_example("../../etc/passwd")
    assert bad["ok"] is False


def test_example_guides_point_to_real_tools_and_columns(tmp_path, monkeypatch):
    from scripts.generate_lite_toolbox_examples import make_examples
    from scripts.generate_lite_stress_examples import make_stress_examples

    out = tmp_path / "tool_benchmarks"
    stress = tmp_path / "tool_stress"
    make_examples(out)
    make_stress_examples(stress)
    monkeypatch.setenv("SCOPE_ANALYZER_EXAMPLES", str(out))
    monkeypatch.setenv("SCOPE_ANALYZER_STRESS_EXAMPLES", str(stress))
    api = Api()
    tools = {t["id"] for t in api.list_tools()["tools"]}
    examples = api.list_examples()["examples"]

    for ex in examples:
        guide = ex["guide"]
        assert guide["tool"] in tools
        loaded = api.load_example(ex["file"])
        assert loaded["ok"] is True
        assert guide["column"] in loaded["y_cols"], ex["file"]


def test_native_clipboard_bridge_contract(monkeypatch):
    calls = {}

    def fake_text(text):
        calls["text"] = text
        return True, "fake text clipboard"

    def fake_image(data, mime):
        calls["image"] = (data, mime)
        return True, "fake image clipboard"

    monkeypatch.setattr(backend_api, "_copy_text_to_clipboard_native", fake_text)
    monkeypatch.setattr(backend_api, "_copy_image_to_clipboard_native", fake_image)

    api = Api()
    text = api.copy_text_to_clipboard("<svg></svg>")
    assert text["ok"] is True
    assert text["read_only"] is True
    assert calls["text"] == "<svg></svg>"

    image = api.copy_image_to_clipboard("data:image/png;base64,QUJDRA==")
    assert image["ok"] is True
    assert image["mime"] == "image/png"
    assert calls["image"] == (b"ABCD", "image/png")

    bad = api.copy_image_to_clipboard("not-a-data-url")
    assert bad["ok"] is False
    assert bad["read_only"] is True


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


def test_pick_csv_filter_descriptions_are_valid_for_pywebview():
    """Regression: pywebview's macOS file dialog only allows word chars and
    spaces in a filter's description (a '/' raised 'not a valid file filter')."""
    import re

    class FakeWindow:
        def create_file_dialog(self, _dialog_type, **kwargs):
            self.kwargs = kwargs
            return []

    fw = FakeWindow()
    api = Api()
    api.set_window(fw)
    api.pick_csv()
    for ft in fw.kwargs["file_types"]:
        desc = ft.split("(")[0].strip()
        assert re.fullmatch(r"[\w ]+", desc), f"illegal filter description: {ft!r}"


def test_pick_csv_uses_valid_filters_no_fallback(tek_csv):
    """Open-dialog filters must satisfy pywebview's own parser, or the packaged
    app shows 'not a valid file filter'. Fails if a filter string regresses."""
    parse = pytest.importorskip("webview.util").parse_file_type
    calls = []

    class FakeWindow:
        def create_file_dialog(self, _dialog_type, **kwargs):
            calls.append(kwargs)
            for ft in kwargs.get("file_types", ()) or ():
                parse(ft)  # raises ValueError if invalid -> pick_csv would fall back
            return [str(tek_csv)]

    api = Api()
    api.set_window(FakeWindow())
    r = api.pick_csv()
    assert r["ok"] is True
    assert len(calls) == 1                        # valid filters -> no unfiltered fallback
    assert calls[0].get("file_types") is not None
