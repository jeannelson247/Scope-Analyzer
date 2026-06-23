from pathlib import Path

from recursive_agent_protocol import (
    build_packet,
    classify_changed_files,
    compact_lines,
    role_prompt,
)


def test_classifies_protected_scientific_core():
    groups = classify_changed_files(["rlc_reconstruct.py", "scope_web/index.html", "tests/test_x.py"])

    assert groups["protected"] == ["rlc_reconstruct.py"]
    assert groups["ui"] == ["scope_web/index.html"]
    assert groups["tests"] == ["tests/test_x.py"]


def test_compact_lines_limits_items_and_chars():
    lines = ["alpha " * 20, "beta " * 20, "gamma " * 20]
    compacted = compact_lines(lines, max_items=2, max_chars=50)

    assert len(compacted) <= 2
    assert sum(len(x) for x in compacted) <= 52


def test_role_prompt_contains_guardrails_and_tests(tmp_path: Path):
    packet = build_packet(
        tmp_path,
        "Audit Lite tool coverage.",
        state_summary=["diagnostics pass"],
        decisions=["do not edit core"],
    )

    prompt = role_prompt(packet, "planner")

    assert "Never overwrite or modify original CSV/TXT source files." in prompt
    assert "python -m pytest -q" in prompt
    assert "Audit Lite tool coverage." in prompt
