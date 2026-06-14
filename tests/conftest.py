"""
Shared fixtures for the Scope Studio engine test suite.

These tests exercise the *deterministic NumPy engine* only — the layer the
project's design rule says must produce every number (the LLM only routes
to it and interprets its output). Nothing here imports Qt, so the suite
runs headless in CI.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest


@pytest.fixture
def tek_csv(tmp_path):
    """Write a small Tektronix-style CSV (preamble + Vertical Units + two
    channels) and return its path. CH2 is a clean linear ramp so tests can
    assert exact, known values."""
    n = 200
    t = np.linspace(-1e-3, 1e-3, n)            # seconds, with pre-trigger
    ch1 = np.linspace(0.0, 2.0, n)             # volts
    ch2 = np.linspace(0.0, 1000.0, n)          # amps (clean ramp)
    lines = [
        "Model,DPO2024B",
        "Vertical Units,V,A",
        "Source,CH1,CH2",
        "TIME,CH1,CH2",
    ]
    for i in range(n):
        lines.append(f"{t[i]:.9e},{ch1[i]:.6e},{ch2[i]:.6e}")
    p = tmp_path / "T0000.CSV"
    p.write_text("\n".join(lines) + "\n")
    return p


@pytest.fixture
def ramp_axes():
    """A shared (x_ms, t_s) axis pair plus a clean ramp signal."""
    n = 1000
    t_s = np.linspace(0.0, 0.05, n)            # 0..50 ms
    x = np.linspace(0.0, 100.0, n)
    return x, t_s
