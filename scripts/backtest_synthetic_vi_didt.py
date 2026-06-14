#!/usr/bin/env python3
"""Backtest synthetic V/I/di-dt files and raw-file immutability."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from csv_loader import load_csv  # noqa: E402
from data_session import DataSession, file_sha256  # noqa: E402
from synthetic_vi_didt import L_INTERNAL_H, write_examples  # noqa: E402


def main() -> int:
    examples = os.path.join(ROOT, "examples")
    written = write_examples(examples)
    scope_path = os.path.join(examples, "synthetic_vi_didt_scope.csv")
    before = file_sha256(scope_path)
    session = DataSession.from_path(scope_path)
    loaded = load_csv(scope_path)
    after_load = file_sha256(scope_path)
    assert before == after_load == session.source_hash

    didt = loaded.df["CH4 dI/dt (A/s)"].to_numpy(float)
    ldidt = loaded.df["CH5 L*dI/dt (V)"].to_numpy(float)
    err = np.nanmax(np.abs(ldidt - L_INTERNAL_H * didt))
    # Text CSV round-tripping introduces microvolt-scale floating error.
    assert err < 1e-4, err

    for name in (
        "synthetic_vi_didt_surface_current.csv",
        "synthetic_vi_didt_surface_didt.csv",
        "synthetic_vi_didt_surface_voltage.csv",
    ):
        df = pd.read_csv(os.path.join(examples, name))
        assert {"x", "y", "z", "c"} <= set(df.columns)
        z = df.pivot_table(index="y", columns="x", values="z",
                           aggfunc="mean")
        c = df.pivot_table(index="y", columns="x", values="c",
                           aggfunc="mean")
        assert z.shape == c.shape
        assert z.notna().to_numpy().mean() > 0.99
        assert z.size > 1000

    print("synthetic V/I/di-dt backtest passed")
    print(f"files: {len(written)}")
    print(f"scope rows: {loaded.n_rows:,}")
    print(f"max |L*dI/dt - V| error: {err:.3e} V")
    print(f"source hash unchanged: {session.short_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
