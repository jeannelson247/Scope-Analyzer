#!/usr/bin/env python3
"""Generate deterministic 166 uH V/I/di-dt examples for Scope Studio."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from synthetic_vi_didt import write_examples  # noqa: E402


def main() -> int:
    out_dir = os.path.join(ROOT, "examples")
    for path in write_examples(out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
