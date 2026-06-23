#!/usr/bin/env python3
"""CLI wrapper for scope_agent_turn.py from the repository root."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scope_agent_turn import main


if __name__ == "__main__":
    raise SystemExit(main())
