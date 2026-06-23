"""RecursiveMAS-inspired agent handoff packets for Scope Studio.

This is intentionally text/JSON based. The real RecursiveMAS paper transfers
latent states through trained adapters; Scope Studio needs a lightweight,
auditable protocol that works with Hermes, MLX-LM, and ordinary local models.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import subprocess
from typing import Iterable, Literal


Role = Literal["planner", "coder", "reviewer", "summarizer"]
Phase = Literal["plan", "implement", "review", "handoff"]


PROTECTED_SCIENCE_MODULES = (
    "csv_loader.py",
    "signal_tools.py",
    "calibration.py",
    "saturation_recovery.py",
    "rlc_reconstruct.py",
    "reconstruction_audit.py",
    "data_quality.py",
    "detect_anomalies.py",
)

DEFAULT_GUARDRAILS = (
    "Never overwrite or modify original CSV/TXT source files.",
    "Do not push to GitHub or any remote without explicit human approval.",
    "Do not rewrite protected scientific-core modules unless the task explicitly approves it.",
    "All filters, formulas, reconstructions, and overlays are in-memory display transforms by default.",
    "Derived CSVs, figures, reports, and logs must be written as new files.",
    "Run tests after edits and report exact commands plus pass/fail status.",
)

DEFAULT_VERIFICATION_COMMANDS = (
    "python -m py_compile ai_assistant.py scope_web/backend_api.py app.py",
    "python -m pytest -q",
    "python scripts/benchmark_lite_toolbox.py",
    "python scripts/benchmark_lite_stress_tools.py",
)


@dataclass
class AgentPacket:
    """Compact task state passed between local agents."""

    project_root: str
    task: str
    phase: Phase = "plan"
    round_index: int = 0
    max_rounds: int = 3
    state_summary: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    protected_files: list[str] = field(default_factory=lambda: list(PROTECTED_SCIENCE_MODULES))
    guardrails: list[str] = field(default_factory=lambda: list(DEFAULT_GUARDRAILS))
    verification_commands: list[str] = field(default_factory=lambda: list(DEFAULT_VERIFICATION_COMMANDS))
    open_questions: list[str] = field(default_factory=list)
    next_action: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _run_git(root: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def git_changed_files(project_root: str | Path) -> list[str]:
    """Return changed paths relative to project root, if git is available."""
    root = Path(project_root)
    out = _run_git(root, ["status", "--short"])
    paths: list[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        # Handles ordinary, renamed, and untracked entries well enough for prompts.
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        paths.append(path)
    return paths


def classify_changed_files(paths: Iterable[str]) -> dict[str, list[str]]:
    """Separate protected scientific files from UI/docs/tests/other files."""
    protected = set(PROTECTED_SCIENCE_MODULES)
    groups = {"protected": [], "ui": [], "tests": [], "docs": [], "other": []}
    for path in paths:
        name = Path(path).name
        if name in protected:
            groups["protected"].append(path)
        elif path.startswith("scope_web/") or path.endswith((".html", ".css", ".qss")):
            groups["ui"].append(path)
        elif path.startswith("tests/") or path.startswith("scripts/benchmark"):
            groups["tests"].append(path)
        elif path.endswith((".md", ".txt")) or path.startswith("docs/"):
            groups["docs"].append(path)
        else:
            groups["other"].append(path)
    return groups


def compact_lines(lines: Iterable[str], max_items: int = 8, max_chars: int = 900) -> list[str]:
    """Keep handoffs short enough for small local models."""
    out: list[str] = []
    total = 0
    for raw in lines:
        line = " ".join(str(raw).split())
        if not line:
            continue
        room = max_chars - total
        if room <= 0 or len(out) >= max_items:
            break
        if len(line) > room:
            line = line[: max(0, room - 3)].rstrip() + "..."
        out.append(line)
        total += len(line) + 1
    return out


def build_packet(
    project_root: str | Path,
    task: str,
    *,
    phase: Phase = "plan",
    round_index: int = 0,
    max_rounds: int = 3,
    state_summary: Iterable[str] = (),
    decisions: Iterable[str] = (),
    open_questions: Iterable[str] = (),
    next_action: str = "",
) -> AgentPacket:
    root = Path(project_root).expanduser().resolve()
    changed = git_changed_files(root)
    return AgentPacket(
        project_root=str(root),
        task=task.strip(),
        phase=phase,
        round_index=round_index,
        max_rounds=max_rounds,
        state_summary=compact_lines(state_summary),
        decisions=compact_lines(decisions),
        changed_files=changed,
        open_questions=compact_lines(open_questions, max_items=5, max_chars=500),
        next_action=" ".join(next_action.split()),
    )


def next_round(
    packet: AgentPacket,
    *,
    phase: Phase,
    observations: Iterable[str] = (),
    decisions: Iterable[str] = (),
    next_action: str = "",
) -> AgentPacket:
    """Create the next compact packet after a model finishes its turn."""
    return AgentPacket(
        project_root=packet.project_root,
        task=packet.task,
        phase=phase,
        round_index=min(packet.round_index + 1, packet.max_rounds),
        max_rounds=packet.max_rounds,
        state_summary=compact_lines([*packet.state_summary, *observations], max_items=10, max_chars=1100),
        decisions=compact_lines([*packet.decisions, *decisions], max_items=10, max_chars=1100),
        changed_files=git_changed_files(packet.project_root),
        open_questions=packet.open_questions,
        next_action=" ".join(next_action.split()),
    )


def role_prompt(packet: AgentPacket, role: Role, *, max_chars: int = 6000) -> str:
    """Render a compact prompt for one local agent role."""
    groups = classify_changed_files(packet.changed_files)
    role_rules = {
        "planner": (
            "Inspect and plan only. Do not edit files. Output a short plan, risk notes, "
            "and exact tests the coder should run."
        ),
        "coder": (
            "Implement only the approved plan. Keep edits minimal. Do not touch protected "
            "scientific files unless the task explicitly allows it."
        ),
        "reviewer": (
            "Review the diff, logic, tests, and guardrails. Prefer finding defects over "
            "adding new code. If unsafe, stop and explain."
        ),
        "summarizer": (
            "Compress the result for the next recursion round. Keep only decisions, changed "
            "files, test results, blockers, and next action."
        ),
    }
    parts = [
        f"# Scope Studio recursive agent packet ({role})",
        f"phase: {packet.phase}",
        f"round: {packet.round_index}/{packet.max_rounds}",
        f"project_root: {packet.project_root}",
        "",
        "## Task",
        packet.task,
        "",
        "## Role rule",
        role_rules[role],
        "",
        "## Guardrails",
        *[f"- {g}" for g in packet.guardrails],
        "",
        "## Protected scientific core",
        *[f"- {p}" for p in packet.protected_files],
        "",
        "## Compact state summary",
        *([f"- {s}" for s in packet.state_summary] or ["- none yet"]),
        "",
        "## Decisions so far",
        *([f"- {d}" for d in packet.decisions] or ["- none yet"]),
        "",
        "## Changed files by category",
        f"- protected: {groups['protected'] or 'none'}",
        f"- ui: {groups['ui'] or 'none'}",
        f"- tests: {groups['tests'] or 'none'}",
        f"- docs: {groups['docs'] or 'none'}",
        f"- other: {groups['other'] or 'none'}",
        "",
        "## Verification commands",
        *[f"- `{cmd}`" for cmd in packet.verification_commands],
        "",
        "## Open questions",
        *([f"- {q}" for q in packet.open_questions] or ["- none"]),
        "",
        "## Next action",
        packet.next_action or "Follow the role rule and produce the smallest useful output.",
        "",
        "## Required response format",
        "Return Markdown with: Summary, Proposed/Completed Changes, Risks, Tests, Next Packet Notes.",
    ]
    text = "\n".join(parts)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 90].rstrip() + "\n\n[truncated: packet exceeded local-model prompt budget]"
