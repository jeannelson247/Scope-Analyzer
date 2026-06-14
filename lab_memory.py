"""
lab_memory.py - the self-tailoring lab assistant's memory.

Design (and honest framing): this is CONTEXT LEARNING, not weight
training. Knowledge accumulates in an append-only journal
(lab_memory/entries.jsonl - never auto-deleted, human-auditable), and a
small LLM-compressed digest (lab_memory/LAB_MEMORY.md) is prepended to
every side-chat prompt. As entries grow, the digest is re-compressed by
the local model on request, so the assistant's working knowledge of YOUR
rig, conventions, and findings grows while staying within context
limits. Patterns "extrapolate" because the digest carries them into
every future conversation.

Why not fine-tuning: a digest is instantly updatable, reviewable line by
line, and cannot corrupt the model. If a stable behavioral style is ever
wanted, the journal doubles as a curated dataset for an mlx-lm LoRA -
that door stays open.

Rules:
  * entries.jsonl is source of truth; the digest is derived and marked
    machine-generated.
  * The digest may summarize but never invent; the rebuild prompt
    forbids adding facts not present in entries.
  * Nothing here touches measurement CSVs.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
MEM_DIR = os.path.join(ROOT, "lab_memory")
ENTRIES = os.path.join(MEM_DIR, "entries.jsonl")
DIGEST = os.path.join(MEM_DIR, "LAB_MEMORY.md")

DIGEST_CHAR_BUDGET = 6000          # ~1.5k tokens carried per prompt

REBUILD_PROMPT = (
    "You maintain the lab-memory digest for a scope-data analysis "
    "assistant. Below are journal entries (newest last). Rewrite the "
    "digest: group related facts (rig constants, sensor behaviors, "
    "calibration values, user preferences, recurring findings), keep "
    "every number with its units, prefer newer corrections over older "
    "statements, and stay under {budget} characters. STRICT RULE: do "
    "not add any fact that is not in the entries.\n\nENTRIES:\n{entries}"
    "\n\nReply with only the digest text."
)


def ensure_dir() -> None:
    os.makedirs(MEM_DIR, exist_ok=True)


def add_entry(text: str, source: str = "user") -> int:
    """Append one knowledge entry; returns total entry count."""
    ensure_dir()
    rec = {"t": datetime.now(timezone.utc).isoformat(),
           "source": source, "text": text.strip()}
    with open(ENTRIES, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return count_entries()


def count_entries() -> int:
    if not os.path.exists(ENTRIES):
        return 0
    with open(ENTRIES, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def all_entries() -> list[dict]:
    if not os.path.exists(ENTRIES):
        return []
    out = []
    with open(ENTRIES, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def digest_text() -> str:
    """Current digest ('' if none yet)."""
    if not os.path.exists(DIGEST):
        return ""
    with open(DIGEST, encoding="utf-8") as fh:
        return fh.read().strip()


def context_block() -> str:
    """What the side chat prepends to prompts (budgeted)."""
    d = digest_text()
    if not d:
        return ""
    return ("Lab memory (user-curated knowledge about this rig and "
            "workflow):\n" + d[:DIGEST_CHAR_BUDGET])


def rebuild_digest(ask_fn) -> str:
    """Re-compress all entries into the digest using the provided
    ask_fn(prompt) -> str (the app passes its current local model).
    Returns the new digest text."""
    ensure_dir()
    entries = all_entries()
    if not entries:
        return ""
    body = "\n".join(f"- [{e['t'][:10]} {e['source']}] {e['text']}"
                     for e in entries)
    reply = ask_fn(REBUILD_PROMPT.format(budget=DIGEST_CHAR_BUDGET,
                                         entries=body[-30000:]))
    digest = (reply or "").strip()
    if not digest:
        return digest_text()
    header = ("<!-- machine-generated digest; entries.jsonl is the "
              "source of truth; rebuild via chat: 'compress memory' -->\n")
    with open(DIGEST, "w", encoding="utf-8") as fh:
        fh.write(header + digest + "\n")
    return digest
