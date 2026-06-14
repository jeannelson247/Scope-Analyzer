"""
tool_sandbox.py - draft-tool workflow for the local Scope Studio assistant.

Draft tools live in tool_sandbox/drafts and are deliberately NOT part of the
approved runtime registry. The LLM can help draft a tool, but a human must
review, test, and promote deterministic code before it can run from the app.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
import re
import textwrap


ROOT = os.path.dirname(os.path.abspath(__file__))
SANDBOX_DIR = os.path.join(ROOT, "tool_sandbox")
DRAFTS_DIR = os.path.join(SANDBOX_DIR, "drafts")


@dataclass(frozen=True)
class DraftToolPaths:
    folder: str
    script: str
    manifest: str
    test_script: str


def _slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_ -]+", "", text).strip().lower()
    text = re.sub(r"[\s-]+", "_", text)
    return text or "draft_tool"


def ensure_sandbox() -> str:
    os.makedirs(DRAFTS_DIR, exist_ok=True)
    readme = os.path.join(SANDBOX_DIR, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as handle:
            handle.write(
                "# Scope Studio Tool Sandbox\n\n"
                "Draft tools created by the local assistant live in "
                "`drafts/`. They are not imported by the app and cannot "
                "touch original CSV files. Promote only reviewed tools with "
                "deterministic tests.\n"
            )
    return SANDBOX_DIR


def create_draft_tool(
    name: str = "moving_average_demo",
    purpose: str = "Example draft tool for in-memory waveform analysis.",
) -> DraftToolPaths:
    ensure_sandbox()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(DRAFTS_DIR, f"{stamp}_{_slug(name)}")
    os.makedirs(folder, exist_ok=False)

    script = os.path.join(folder, "tool.py")
    manifest = os.path.join(folder, "manifest.json")
    test_script = os.path.join(folder, "test_tool.py")

    manifest_obj = {
        "name": name,
        "status": "draft_not_approved",
        "purpose": purpose,
        "inputs": ["time_array", "channel_arrays"],
        "outputs": ["text_report", "optional_overlay_arrays"],
        "required_packages": ["numpy"],
        "safety": [
            "Never modifies original CSV files.",
            "Runs only on in-memory arrays passed by Scope Studio.",
            "Must pass deterministic tests before promotion.",
        ],
    }
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest_obj, handle, indent=2)
        handle.write("\n")

    with open(script, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(
            '''\
            """
            Draft Scope Studio tool.

            This file is intentionally inactive. Review, test, and promote it
            into the approved tool registry before using it in the app.
            """
            from __future__ import annotations

            import numpy as np


            def run(time_array, channel_arrays, **kwargs):
                """Return a deterministic report from in-memory arrays only."""
                t = np.asarray(time_array, dtype=float)
                if t.size == 0:
                    return {"text": "No samples supplied.", "overlays": []}
                lines = [f"Samples: {t.size:,}"]
                for name, values in channel_arrays.items():
                    y = np.asarray(values, dtype=float)
                    lines.append(
                        f"{name}: mean={np.nanmean(y):.6g}, "
                        f"peak={np.nanmax(y):.6g}"
                    )
                return {"text": "\\n".join(lines), "overlays": []}
            '''
        ))

    with open(test_script, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(
            '''\
            from tool import run


            def test_run():
                out = run([0, 1, 2], {"demo": [0, 2, 4]})
                assert "Samples: 3" in out["text"]
                assert "peak=4" in out["text"]


            if __name__ == "__main__":
                test_run()
                print("draft tool self-test passed")
            '''
        ))

    return DraftToolPaths(folder, script, manifest, test_script)
