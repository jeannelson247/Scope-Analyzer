"""Run advanced Scope Analyzer Lite stress examples through deterministic tools.

This is a release-audit companion to ``benchmark_lite_toolbox.py``. The regular
benchmark proves the beginner examples work. This script uses harsher synthetic
data to catch edge cases before a packaged app is shared.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scope_web"))

from backend_api import Api  # noqa: E402
from scripts.generate_lite_stress_examples import DEFAULT_OUT, make_stress_examples  # noqa: E402

BACKTEST_DIR = ROOT / "backtests"
REPORT_PATH = BACKTEST_DIR / "lite_stress_benchmark.txt"
JSON_PATH = BACKTEST_DIR / "lite_stress_benchmark.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def load(api: Api, path: Path) -> dict:
    r = api.load_csv(str(path))
    require(r.get("ok"), f"load failed for {path.name}: {r.get('error')}")
    return r


def tool_ok(api: Api, tool: str, params: dict) -> dict:
    r = api.run_tool(tool, params)
    require(r.get("ok"), f"{tool} failed: {r.get('error') or r.get('text')}")
    require(r.get("read_only") is True, f"{tool} did not report read_only=True")
    return r


def case_01(api: Api, path: Path) -> str:
    r = load(api, path)
    require(r["n_rows"] == 80_000, "large stress dataset row count changed")
    st = tool_ok(api, "stats", {"column": "Current_A"})
    require(st["max"] > 5000, "large current peak unexpectedly low")
    an = tool_ok(api, "anomaly", {"column": "Current_A", "threshold_sigma": 6})
    require("Anomaly" in an["text"], "anomaly report missing")
    return "large trace loads, decimates, and anomaly scan completes"


def case_02(api: Api, path: Path) -> str:
    load(api, path)
    qc = tool_ok(api, "quality", {"column": "Signal_V"})
    require(qc.get("status") in {"warning", "error"}, "QC did not flag bad timing/NaN")
    return f"QC flags nonuniform timestamp case ({qc.get('status')})"


def case_03(api: Api, path: Path) -> str:
    load(api, path)
    fft = tool_ok(api, "fft", {"column": "Signal_V", "f_min": 1000})
    dom = float(fft["dominant_frequency_hz"])
    require(abs(dom - 42_000) < 1_500, f"dominant frequency {dom:g} not near 42 kHz")
    return f"FFT finds fixed 42 kHz spur ({dom:.1f} Hz)"


def case_04(api: Api, path: Path) -> str:
    load(api, path)
    lp = tool_ok(api, "lowpass", {"column": "Noisy_current_A", "cutoff_hz": 15_000})
    require(lp.get("series"), "low-pass returned no series")
    fft = tool_ok(api, "fft", {"column": "Noisy_current_A", "f_min": 10_000})
    require(float(fft["dominant_frequency_hz"]) > 100_000, "ringing spur not present")
    return "low-pass survives impulse/ringing stress"


def case_05(api: Api, path: Path) -> str:
    load(api, path)
    formula = api.run_tool("formula", {"column": "Sensor_V", "formula": "(x-2.5)*750",
                                       "label": "Sensor_V_to_A", "unit": "A"})
    require(formula.get("ok"), f"formula failed: {formula.get('error')}")
    cal = tool_ok(api, "calibration", {"source": "Source_A_uncal", "reference": "Reference_A",
                                       "t_start": 0.010, "t_end": 0.075})
    require(1.33 < float(cal["slope"]) < 1.45, f"calibration slope {cal['slope']:g} outside expected")
    return f"formula and drifted calibration pass (slope {cal['slope']:.4f})"


def case_06(api: Api, path: Path) -> str:
    load(api, path)
    sat = tool_ok(api, "saturation", {"column": "BBCM_A", "sat_level": 6000})
    require(sat.get("overlay"), "saturation overlay missing")
    rlc = tool_ok(api, "rlc", {"column": "BBCM_A", "sat_level": 6000,
                               "ref_channel": "Pearson_A", "ref_start": 0,
                               "ref_end": 0.006, "trusted_windows": "0:0.006, 0.050:0.150"})
    peak = float(rlc["params"]["peak"])
    require(6500 < peak < 9500, f"hidden peak {peak:g} outside expected")
    return f"multi-window hidden peak reconstruction pass ({peak:.0f} A)"


def case_07(api: Api, path: Path) -> str:
    load(api, path)
    st = tool_ok(api, "stats", {"column": "Current_A"})
    require(st["max"] > 1500 and st["min"] < -500, "bipolar extrema not preserved")
    integ = tool_ok(api, "integrate", {"column": "Current_A"})
    require(integ.get("series"), "integral returned no series")
    return "bipolar stats and integration preserve sign"


def case_08(api: Api, path: Path) -> str:
    load(api, path)
    qc = tool_ok(api, "quality", {"column": "Current_A"})
    require(qc.get("status") in {"warning", "error"}, "QC missed dropout/NaN")
    an = tool_ok(api, "anomaly", {"column": "Current_A", "threshold_sigma": 5})
    require("Anomaly" in an["text"], "anomaly report missing")
    require("flatline/dropout" in an["text"], "flatline/dropout report missing")
    return "flatline/dropout quality and anomaly tools complete"


def case_09(api: Api, path: Path) -> str:
    load(api, path)
    peaks = [tool_ok(api, "stats", {"column": f"Module{i}_A"})["max"] for i in range(1, 5)]
    require(peaks[2] == max(peaks), "Module3 should be the high module")
    return "module-skew statistics expose the high channel"


def case_10(api: Api, path: Path) -> str:
    load(api, path)
    grad = tool_ok(api, "gradient", {"column": "Current_A"})
    require(grad.get("series"), "gradient returned no series")
    integ = tool_ok(api, "integrate", {"column": "Rogowski_V"})
    require(integ.get("series"), "integral returned no series")
    return "Rogowski-style derivative/integral tools complete"


def case_11(api: Api, path: Path) -> str:
    load(api, path)
    st_i = tool_ok(api, "stats", {"column": "Current_A"})
    st_v = tool_ok(api, "stats", {"column": "Control_V"})
    require(st_i["max"] / max(abs(st_v["max"]), 1e-12) > 10_000, "dynamic range not large enough")
    return "high dynamic range axes stats pass"


def case_12(api: Api, path: Path) -> str:
    load(api, path)
    grad = tool_ok(api, "gradient", {"column": "Current_A"})
    require(grad.get("series"), "gradient returned no series")
    st = tool_ok(api, "stats", {"column": "L_dIdt_V"})
    require(50 < st["max"] < 160, "L*dI/dt voltage outside expected drive range")
    return "166 uH V/I/dI-dt ripple stress passes"


CASES: dict[str, Callable[[Api, Path], str]] = {
    "stress_01_large_decimation_spikes.csv": case_01,
    "stress_02_nonuniform_time_nan.csv": case_02,
    "stress_03_fft_chirp_spur.csv": case_03,
    "stress_04_filter_impulse_ringing.csv": case_04,
    "stress_05_calibration_drift.csv": case_05,
    "stress_06_censored_multiwindow_6ka.csv": case_06,
    "stress_07_bipolar_return.csv": case_07,
    "stress_08_flatline_dropout.csv": case_08,
    "stress_09_module_skew_noise.csv": case_09,
    "stress_10_rogowski_drift.csv": case_10,
    "stress_11_high_dynamic_range_axes.csv": case_11,
    "stress_12_vi_didt_166uH_ripple.csv": case_12,
}


def run(out_dir: Path = DEFAULT_OUT) -> tuple[int, list[dict[str, object]]]:
    if not (out_dir / "manifest.json").exists():
        make_stress_examples(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))["datasets"]
    results: list[dict[str, object]] = []
    for item in manifest:
        fname = str(item["file"])
        path = out_dir / fname
        api = Api()
        before = sha256(path)
        try:
            detail = CASES[fname](api, path)
            after = sha256(path)
            require(before == after, "source CSV hash changed")
            result = {"file": fname, "title": item["title"], "ok": True,
                      "detail": detail, "sha256": before}
        except Exception as exc:
            result = {"file": fname, "title": item.get("title", ""), "ok": False,
                      "detail": f"{type(exc).__name__}: {exc}", "sha256": before}
        results.append(result)
    return sum(1 for r in results if not r["ok"]), results


def write_report(results: list[dict[str, object]]) -> None:
    BACKTEST_DIR.mkdir(exist_ok=True)
    ok = sum(1 for r in results if r["ok"])
    lines = [
        "Scope Analyzer Lite stress benchmark",
        "====================================",
        f"Datasets: {len(results)} synthetic stress CSV files",
        f"Result: {ok} pass / {len(results) - ok} fail",
        "",
        "Each case loads through scope_web.backend_api.Api, runs deterministic",
        "tools, and verifies that the source CSV hash is unchanged.",
        "",
    ]
    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        lines.append(f"[{mark}] {r['file']} - {r['title']}")
        lines.append(f"       {r['detail']}")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    make_stress_examples(DEFAULT_OUT)
    fails, results = run(DEFAULT_OUT)
    write_report(results)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
