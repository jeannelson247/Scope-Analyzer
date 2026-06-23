#!/usr/bin/env python3
"""Generate compact RecursiveMAS-inspired prompts for local Scope Studio agents."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recursive_agent_protocol import build_packet, role_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--task", required=True)
    parser.add_argument("--role", choices=["planner", "coder", "reviewer", "summarizer"], default="planner")
    parser.add_argument("--phase", choices=["plan", "implement", "review", "handoff"], default="plan")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--state", action="append", default=[], help="Compact state note; may be repeated.")
    parser.add_argument("--decision", action="append", default=[], help="Decision note; may be repeated.")
    parser.add_argument("--question", action="append", default=[], help="Open question; may be repeated.")
    parser.add_argument("--next-action", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    packet = build_packet(
        args.project_root,
        args.task,
        phase=args.phase,
        round_index=args.round,
        max_rounds=args.max_rounds,
        state_summary=args.state,
        decisions=args.decision,
        open_questions=args.question,
        next_action=args.next_action,
    )
    prompt = role_prompt(packet, args.role)

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{stamp}_{args.role}_round{args.round}"
        (out_dir / f"{stem}.json").write_text(packet.to_json(), encoding="utf-8")
        (out_dir / f"{stem}.md").write_text(prompt, encoding="utf-8")
        print(out_dir / f"{stem}.md")
        print(out_dir / f"{stem}.json")
    else:
        print(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
