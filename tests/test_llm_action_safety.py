"""
Contract tests for the local-LLM action boundary.

Scope Studio's release claim is that the LLM routes and explains, while
deterministic tools compute. These tests make that claim hard to regress:
unknown tool names do not execute, malformed JSON is ignored, the router
schema remains a small allowlist, and even a malicious formula action is
only a visible field edit that the formula sandbox rejects at evaluation.
"""
from __future__ import annotations

import numpy as np
import pytest

import ai_assistant
import chat_actions
import signal_tools


APPROVED_ROUTER_RUNS = {
    "detect_anomalies",
    "channel_stats",
    "estimate_saturation",
    "reconstruct_rlc",
    "rlc_audit",
    "zero_baseline",
    "none",
}


class FakeCache:
    def __init__(self) -> None:
        self.cleared = False

    def clear(self) -> None:
        self.cleared = True


class FakeChannel:
    def __init__(self, name: str = "CH1") -> None:
        self.name = name
        self.enabled = False
        self.axis = "left"
        self.gain = 1.0
        self.offset = 0.0
        self.label = ""
        self.formula = "x"

    def display_label(self) -> str:
        return self.label or self.name


class FakeWindow:
    def __init__(self) -> None:
        self.channels = [FakeChannel()]
        self._transform_cache = FakeCache()
        self.rebuilt = False
        self.refreshed = False
        self.undo_labels: list[str] = []

    def push_display_undo(self, label: str) -> None:
        self.undo_labels.append(label)

    def _rebuild_table(self) -> None:
        self.rebuilt = True

    def refresh_plot(self) -> None:
        self.refreshed = True


def test_extract_actions_ignores_malformed_json() -> None:
    reply = "Sure.\n```json\n{\"actions\": [}\n```\nStill text."
    clean, actions = chat_actions.extract_actions(reply)
    assert actions == []
    assert "```json" in clean


def test_extract_actions_strips_valid_action_block() -> None:
    reply = 'Done.\n```json\n{"actions": [{"run": "detect_anomalies"}]}\n```'
    clean, actions = chat_actions.extract_actions(reply)
    assert clean == "Done."
    assert actions == [{"run": "detect_anomalies"}]


def test_unknown_tool_name_is_rejected_not_executed() -> None:
    msg = chat_actions.run_tool(FakeWindow(), {"run": "__import__('os').system('echo pwn')"})
    assert msg.startswith("Unknown tool:")


def test_router_schema_is_small_deterministic_allowlist() -> None:
    run_schema = ai_assistant.ACTION_SCHEMA["properties"]["run"]
    assert set(run_schema["enum"]) == APPROVED_ROUTER_RUNS
    forbidden = {"python", "eval", "exec", "shell", "subprocess", "open_url"}
    assert forbidden.isdisjoint(run_schema["enum"])


def test_channel_formula_action_still_goes_through_formula_sandbox() -> None:
    win = FakeWindow()
    malicious = "__import__('os').system('echo pwn')"
    applied, tool_msgs = chat_actions.apply_actions(
        win,
        [{"channel": {"name": "CH1", "enabled": True, "formula": malicious}}],
    )

    assert tool_msgs == []
    assert "channel CH1 updated" in applied
    assert win.rebuilt is True
    assert win.refreshed is True
    assert win.channels[0].formula == malicious

    x = np.arange(8, dtype=np.float64)
    t = np.linspace(0.0, 1.0, x.size)
    with pytest.raises(signal_tools.FormulaError):
        signal_tools.evaluate_formula(win.channels[0].formula, x, t)


def test_numeric_answer_is_not_accepted_as_a_tool_action() -> None:
    """A model may write prose with numbers, but numbers alone do not trigger
    tool execution. Only an explicit JSON action block crosses the boundary."""
    clean, actions = chat_actions.extract_actions("The peak is probably 6600 A.")
    assert clean == "The peak is probably 6600 A."
    assert actions == []
