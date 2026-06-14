"""
obsidian_notes.py - connected lab notes in an Obsidian vault.

An Obsidian vault is just a folder of Markdown files; [[wikilinks]]
create the graph. Scope Studio writes one note per analysis session
into <vault>/ScopeStudio/, linking shots, tools, the rig, and the
campaign date - so the knowledge graph grows as you work, and the
"which tools ran, and why" report the user wants is generated
deterministically from the session history (never invented by a model).

Vault location is remembered in scope_studio_user_profile.json
(gitignored). Nothing here touches measurement CSVs.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.path.join(ROOT, "scope_studio_user_profile.json")

# deterministic "why" for the tools-used report (kept next to the tool
# registry on purpose: update both when adding a tool)
TOOL_WHY = {
    "compute_stats": "peak/min/mean/RMS over the visible window - the "
                     "baseline quantitative description of the shot",
    "channel_stats": "peak/min/mean/RMS over the visible window",
    "detect_anomalies": "robust (MAD) spike/clipping/drift/imbalance scan "
                        "- separates real events from EMI before any "
                        "interpretation",
    "zero_baseline": "removes pre-trigger offsets so both monitors share "
                     "a true 0 A reference",
    "estimate_saturation": "two-slope censored fit - quick true-peak "
                           "estimate where a monitor exceeded its range",
    "reconstruct_rlc": "censored-ML overdamped-RLC fit - full waveform "
                       "through the censored region, consistent with all "
                       "valid sensor data",
    "lowpass_filter": "suppresses switching noise above the cutoff for "
                      "display and slope estimates",
    "moving_average": "local smoothing for trend visibility",
}


def get_vault() -> str | None:
    try:
        with open(PROFILE, encoding="utf-8") as fh:
            return json.load(fh).get("obsidian_vault")
    except Exception:
        return None


def set_vault(path: str) -> None:
    prof = {}
    try:
        with open(PROFILE, encoding="utf-8") as fh:
            prof = json.load(fh)
    except Exception:
        pass
    prof["obsidian_vault"] = path
    with open(PROFILE, "w", encoding="utf-8") as fh:
        json.dump(prof, fh, indent=2)


def _slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_ -]+", "", text).strip()
    return re.sub(r"\s+", " ", text) or "note"


def session_note_markdown(shot_name: str, source_path: str,
                          source_hash: str, channels: list[str],
                          tool_events: list[str],
                          ai_events: list[str] | None = None,
                          user_comment: str = "") -> str:
    """Build the connected session note. tool_events are the raw
    '[Step] ...' / tool texts from the chat history - quoted verbatim
    so the note is a faithful record, not a paraphrase."""
    today = datetime.now().strftime("%Y-%m-%d")
    used = []
    for name, why in TOOL_WHY.items():
        words = name.split("_")
        if any(name in ev
               or all(w in ev.lower() for w in words)
               for ev in tool_events):
            used.append((name, why))
    lines = [
        f"# Shot {shot_name} - analysis session",
        "",
        f"date:: [[{today}]]",
        f"rig:: [[Tokamak TF current drivers]]",
        f"app:: [[Scope Studio]]",
        f"source:: `{os.path.basename(source_path)}`",
        f"sha256:: `{source_hash[:16]}…`  (original file read-only)",
        f"channels:: " + ", ".join(f"[[channel {c}]]" for c in channels),
        "",
        "## Tools used, and why",
        "",
    ]
    if used:
        for name, why in used:
            lines.append(f"- [[tool {name}]] — {why}")
    else:
        lines.append("- (no deterministic tools were run this session)")
    lines += ["", "## Results (verbatim tool output)", ""]
    for ev in tool_events[-12:]:
        lines.append("> " + ev.replace("\n", "\n> "))
        lines.append("")
    if ai_events:
        lines += ["## AI annotation trace", ""]
        for ev in ai_events[-12:]:
            lines.append(f"- {ev}")
        lines.append("")
    if user_comment:
        lines += ["## My interpretation", "", user_comment, ""]
    lines += ["## Open questions", "", "- ", ""]
    return "\n".join(lines)


def write_note(vault: str, title: str, markdown: str) -> str:
    folder = os.path.join(vault, "ScopeStudio")
    os.makedirs(folder, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H%M")
    path = os.path.join(folder, f"{stamp} {_slug(title)}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    return path
