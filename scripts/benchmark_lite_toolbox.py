"""Benchmark Scope Analyzer Lite deterministic tools against synthetic examples.

This script is intentionally no-LLM. It loads the 15 benchmark CSVs through the
same scope_web.backend_api.Api bridge used by the Lite app and verifies that the
analysis buttons are functional, deterministic, and read-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scope_web"))

from backend_api import Api  # noqa: E402
from scripts.generate_lite_toolbox_examples import DEFAULT_OUT, make_examples  # noqa: E402

BACKTEST_DIR = ROOT / "backtests"
REPORT_PATH = BACKTEST_DIR / "lite_toolbox_benchmark.txt"
JSON_PATH = BACKTEST_DIR / "lite_toolbox_benchmark.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(api: Api, path: Path) -> dict:
    r = api.load_csv(str(path))
    if not r.get("ok"):
        raise AssertionError(f"load failed for {path.name}: {r.get('error')}")
    return r


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def tool_ok(api: Api, tool: str, params: dict) -> dict:
    r = api.run_tool(tool, params)
    require(r.get("ok"), f"{tool} failed: {r.get('error') or r.get('text')}")
    require(r.get("read_only") is True, f"{tool} did not report read_only=True")
    return r


def case_01_stats_gradient_integral_rlc(api: Api, path: Path) -> str:
    load(api, path)
    st = tool_ok(api, "stats", {"column": "Current_A"})
    require(4100 < st["max"] < 4300, "stats peak outside clean pulse range")
    gr = tool_ok(api, "gradient", {"column": "Current_A"})
    require("series" in gr, "gradient returned no derived series")
    integ = tool_ok(api, "integrate", {"column": "Current_A"})
    require(integ["series"]["y"][-1] > 100, "integral/charge too small")
    rlc = tool_ok(api, "rlc", {"column": "Current_A", "t_start": 0.0, "t_end": 0.115})
    require(rlc.get("overlay") and rlc.get("params"), "RLC returned no overlay/params")
    return "stats, gradient, integral, RLC overlays OK"


def case_02_hidden_peak(api: Api, path: Path) -> str:
    load(api, path)
    sat = tool_ok(api, "saturation", {"column": "BBCM_A", "sat_level": 6000})
    require(sat.get("overlay"), "saturation returned no overlay")
    rlc = tool_ok(api, "rlc", {"column": "BBCM_A", "sat_level": 6000,
                               "ref_channel": "Pearson_A", "ref_start": 0.0,
                               "ref_end": 0.010, "trusted_windows": "0:0.005, 0.040:0.150"})
    peak = float(rlc["params"]["peak"])
    require(6500 < peak < 8500, f"RLC peak {peak:g} outside expected hidden-peak range")
    pipe = tool_ok(api, "pipeline", {"column": "BBCM_A", "sat_level": 6000,
                                     "trusted_windows": "0:0.005, 0.040:0.150"})
    require(pipe.get("overlay"), "pipeline did not carry reconstruction overlay")
    return f"saturation + RLC hidden peak OK (peak {peak:.0f} A)"


def case_03_lowpass(api: Api, path: Path) -> str:
    load(api, path)
    lp = tool_ok(api, "lowpass", {"column": "Current_noisy_A", "cutoff_hz": 15000})
    require("series" in lp, "lowpass returned no series")
    an = tool_ok(api, "anomaly", {"column": "Current_noisy_A", "threshold_sigma": 6})
    require("Anomaly scan" in an.get("text", ""), "anomaly text missing")
    return "low-pass and anomaly text OK"


def case_04_fft(api: Api, path: Path) -> str:
    load(api, path)
    fft = tool_ok(api, "fft", {"column": "Signal_V", "f_min": 1000})
    dom = float(fft["dominant_frequency_hz"])
    require(abs(dom - 20000) < 1500, f"dominant frequency {dom:g} not near 20 kHz")
    return f"FFT dominant frequency OK ({dom:.1f} Hz)"


def case_05_formula_calibration(api: Api, path: Path) -> str:
    load(api, path)
    f = api.run_tool("formula", {"column": "Sensor_V", "formula": "(x-2.5)*750",
                                 "label": "Sensor_V_to_A", "unit": "A"})
    require(f.get("ok"), f"formula failed: {f.get('error')}")
    st = tool_ok(api, "stats", {"column": "Sensor_V_to_A"})
    require(1700 < st["max"] < 1900, "formula-converted peak not near reference")
    cal = tool_ok(api, "calibration", {"source": "Sensor_scaled_A", "reference": "Reference_A",
                                       "t_start": 0.006, "t_end": 0.070})
    require(1.18 < cal["slope"] < 1.26, f"calibration slope {cal['slope']:g} outside expected range")
    return f"formula + forced-origin calibration OK (slope {cal['slope']:.4f})"


def case_06_didt(api: Api, path: Path) -> str:
    load(api, path)
    gr = tool_ok(api, "gradient", {"column": "Current_A"})
    require("dI-dt" in gr.get("text", "") or "derivative" in gr.get("text", ""), "gradient text missing")
    st = tool_ok(api, "stats", {"column": "L_dIdt_V"})
    require(st["max"] > 5.0, "L*dI/dt voltage too small")
    return "gradient and L*dI/dt stats OK"


def case_07_integral(api: Api, path: Path) -> str:
    load(api, path)
    integ = tool_ok(api, "integrate", {"column": "Current_A"})
    q = float(integ["series"]["y"][-1])
    require(4.6 < q < 5.4, f"charge integral {q:g} C not near 5 C")
    return f"integral/charge OK ({q:.3f} C)"


def case_08_movmean(api: Api, path: Path) -> str:
    load(api, path)
    sm = tool_ok(api, "movmean", {"column": "Noisy_current_A", "window": 101})
    require("series" in sm, "movmean returned no series")
    return "moving average derived trace OK"


def case_09_anomaly(api: Api, path: Path) -> str:
    load(api, path)
    an = tool_ok(api, "anomaly", {"column": "Current_A", "threshold_sigma": 5})
    text = an.get("text", "").lower()
    require("spike" in text or "crest" in text, "anomaly scan did not flag injected spikes")
    return "anomaly scan catches injected spikes"


def case_10_quality(api: Api, path: Path) -> str:
    load(api, path)
    qc = tool_ok(api, "quality", {"column": "Signal_V"})
    require(qc.get("status") in {"warning", "error"}, "bad CSV quality case was not flagged")
    return f"quality report flags bad timing/data ({qc.get('status')})"


def case_11_baseline_formula(api: Api, path: Path) -> str:
    load(api, path)
    f = api.run_tool("formula", {"column": "Raw_offset_A", "formula": "baseline(x,t,-0.001)",
                                 "label": "Baseline_corrected_A", "unit": "A"})
    require(f.get("ok"), f"baseline formula failed: {f.get('error')}")
    st = tool_ok(api, "stats", {"column": "Baseline_corrected_A", "t_start": -0.010, "t_end": -0.002})
    require(abs(st["mean"]) < 5.0, "baseline-corrected pretrigger mean not near zero")
    return "baseline formula helper OK"


def case_12_soft_saturation(api: Api, path: Path) -> str:
    load(api, path)
    sat = tool_ok(api, "saturation", {"column": "Soft_BBCM_A", "sat_level": 6000})
    require(sat.get("overlay"), "soft saturation returned no overlay")
    rlc = tool_ok(api, "rlc", {"column": "Soft_BBCM_A", "sat_level": 6000,
                               "trusted_windows": "0:0.005, 0.050:0.140"})
    require(rlc.get("params"), "soft saturation RLC params missing")
    return "known-level soft saturation mode OK"


def case_13_modules(api: Api, path: Path) -> str:
    load(api, path)
    peaks = []
    for col in ["Module1_A", "Module2_A", "Module3_A", "Module4_A"]:
        peaks.append(tool_ok(api, "stats", {"column": col})["max"])
    spread = (max(peaks) - min(peaks)) / max(np.mean(peaks), 1e-12)
    require(spread > 0.10, "module-balance example did not contain intended imbalance")
    return f"module stats expose imbalance ({100*spread:.1f}% peak spread)"


def case_14_negative(api: Api, path: Path) -> str:
    load(api, path)
    st = tool_ok(api, "stats", {"column": "Negative_current_A"})
    require(st["min"] < -2500, "negative pulse minimum not detected")
    rlc = tool_ok(api, "rlc", {"column": "Negative_current_A", "t_start": 0.0, "t_end": 0.115})
    require(float(rlc["params"]["peak"]) < 0, "negative-pulse RLC did not preserve sign")
    return "negative-polarity stats/RLC OK"


def case_15_vi_didt(api: Api, path: Path) -> str:
    load(api, path)
    gr = tool_ok(api, "gradient", {"column": "Current_A"})
    require(gr.get("series"), "V/I/dI-dt gradient returned no series")
    st = tool_ok(api, "stats", {"column": "L_dIdt_V"})
    require(40 < st["max"] < 120, "L*dI/dt voltage not in drive-voltage range")
    return "166 uH V/I/dI-dt benchmark OK"


CASES: dict[str, Callable[[Api, Path], str]] = {
    "01_clean_rl_pulse.csv": case_01_stats_gradient_integral_rlc,
    "02_bbcm_clipped_6ka.csv": case_02_hidden_peak,
    "03_lowpass_ringing.csv": case_03_lowpass,
    "04_fft_two_tone.csv": case_04_fft,
    "05_calibration_pair.csv": case_05_formula_calibration,
    "06_didt_voltage_166uH.csv": case_06_didt,
    "07_charge_integral.csv": case_07_integral,
    "08_moving_average_noise.csv": case_08_movmean,
    "09_spikes_anomalies.csv": case_09_anomaly,
    "10_quality_gap_nan_duplicate.csv": case_10_quality,
    "11_baseline_offset.csv": case_11_baseline_formula,
    "12_soft_saturation.csv": case_12_soft_saturation,
    "13_module_balance.csv": case_13_modules,
    "14_negative_pulse.csv": case_14_negative,
    "15_vi_didt_166uH.csv": case_15_vi_didt,
}


def run(out_dir: Path = DEFAULT_OUT) -> tuple[int, list[dict[str, object]]]:
    if not (out_dir / "manifest.json").exists():
        make_examples(out_dir)
    with (out_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)["datasets"]
    results = []
    for item in manifest:
        fname = item["file"]
        path = out_dir / fname
        api = Api()
        before = sha256(path)
        try:
            detail = CASES[fname](api, path)
            after = sha256(path)
            require(before == after, "source CSV hash changed")
            result = {"file": fname, "title": item["title"], "ok": True, "detail": detail,
                      "sha256": before}
        except Exception as exc:
            result = {"file": fname, "title": item.get("title", ""), "ok": False,
                      "detail": f"{type(exc).__name__}: {exc}", "sha256": before}
        results.append(result)
    return sum(1 for r in results if not r["ok"]), results


def write_report(results: list[dict[str, object]]) -> None:
    BACKTEST_DIR.mkdir(exist_ok=True)
    ok = sum(1 for r in results if r["ok"])
    lines = [
        "Scope Analyzer Lite toolbox benchmark",
        "======================================",
        f"Datasets: {len(results)} synthetic CSV files",
        f"Result: {ok} pass / {len(results) - ok} fail",
        "",
        "Each case loads through scope_web.backend_api.Api, runs the relevant",
        "deterministic tool(s), and verifies that the source CSV hash is unchanged.",
        "",
    ]
    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        lines.append(f"[{mark}] {r['file']} - {r['title']}")
        lines.append(f"       {r['detail']}")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    make_examples(DEFAULT_OUT)
    fails, results = run(DEFAULT_OUT)
    write_report(results)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
