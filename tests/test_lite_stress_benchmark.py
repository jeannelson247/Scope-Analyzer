"""Regression coverage for the advanced Lite stress-test pack."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scope_web"))

from scripts.generate_lite_stress_examples import make_stress_examples  # noqa: E402
from scripts.benchmark_lite_stress_tools import run  # noqa: E402


def test_lite_stress_examples_all_pass(tmp_path):
    out_dir = tmp_path / "tool_stress"
    make_stress_examples(out_dir)
    fails, results = run(out_dir)
    assert fails == 0
    assert len(results) == 12
    assert all(r["ok"] for r in results)


def test_stress_pack_guides_cover_real_tools_and_columns(tmp_path, monkeypatch):
    from backend_api import Api

    out_dir = tmp_path / "tool_stress"
    make_stress_examples(out_dir)
    monkeypatch.setenv("SCOPE_ANALYZER_STRESS_EXAMPLES", str(out_dir))
    monkeypatch.setenv("SCOPE_ANALYZER_EXAMPLES", str(tmp_path / "tool_benchmarks"))

    api = Api()
    tools = {t["id"] for t in api.list_tools()["tools"]}
    stress_examples = [
        e for e in api.list_examples()["examples"]
        if e["group"] == "Stress-test datasets"
    ]

    assert len(stress_examples) == 12
    for ex in stress_examples:
        guide = ex["guide"]
        assert guide["tool"] in tools
        loaded = api.load_example(ex["file"])
        assert loaded["ok"] is True
        assert guide["column"] in loaded["y_cols"], ex["file"]
