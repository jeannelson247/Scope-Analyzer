"""Regression coverage for the Lite no-LLM toolbox examples.

The examples are deliberately synthetic and small enough for CI. They prove that
the visible Lite tools are not placeholders: each dataset loads through the same
Python bridge used by the web app, runs the intended deterministic tool, and
leaves the source CSV hash unchanged.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scope_web"))

from backend_api import Api  # noqa: E402
from scripts.generate_lite_toolbox_examples import make_examples  # noqa: E402
from scripts.benchmark_lite_toolbox import run  # noqa: E402


def test_lite_toolbox_examples_all_pass(tmp_path):
    out_dir = tmp_path / "tool_benchmarks"
    make_examples(out_dir)
    fails, results = run(out_dir)
    assert fails == 0
    assert len(results) == 15
    assert all(r["ok"] for r in results)


def test_toolbox_help_is_available_without_llm():
    r = Api().toolbox_help()
    assert r["ok"] is True
    assert r["read_only"] is True
    assert "Scope Analyzer Lite Toolbox FAQ" in r["text"]
    assert "Recover hidden peak" in r["text"]


def test_toolbox_self_check_runs_in_app_bridge(tmp_path, monkeypatch):
    out_dir = tmp_path / "tool_benchmarks"
    make_examples(out_dir)
    monkeypatch.setenv("SCOPE_ANALYZER_EXAMPLES", str(out_dir))

    r = Api().toolbox_self_check()

    assert r["ok"] is True
    assert r["read_only"] is True
    assert r["passes"] == 15
    assert r["fails"] == 0
    assert r["n"] == 15
    assert "15 pass / 0 fail" in r["text"]
    assert "source CSV hash is unchanged" in r["text"]


def test_every_data_tool_has_a_benchmark_example(tmp_path):
    """Guard: every deterministic data tool the bridge exposes is exercised by
    at least one benchmark dataset, so a new tool cannot ship uncovered."""
    manifest = make_examples(tmp_path / "tb")
    covered = set()
    for entry in manifest:
        covered.update(entry.get("tools", []))
    expected = {"stats", "quality", "anomaly", "saturation", "rlc",
                "rlc_audit", "formula", "lowpass", "movmean", "gradient",
                "integrate", "fft", "calibration", "pipeline"}
    missing = expected - covered
    assert not missing, f"tools with no benchmark example: {sorted(missing)}"
