"""
Numerical-accuracy tests for signal_tools' math helpers.

Where the engine smoke tests check that functions run and reject bad input,
these check that the MATH IS RIGHT, against closed-form signals with a known
analytic answer. This protects the formula helpers (the lab's MATLAB-style
conversions) from silent numerical regressions.
"""
from __future__ import annotations

import numpy as np
import pytest

import signal_tools as st


# interior slice: avoid filter/gradient edge transients when comparing
def _interior(a, frac=0.05):
    n = len(a)
    k = int(n * frac)
    return a[k:n - k]


def test_integrate_cos_gives_sin():
    # ∫_0^t cos(τ) dτ = sin(t); trapezoidal on a fine grid should be close.
    t = np.linspace(0.0, 2 * np.pi, 20_000)
    y = np.cos(t)
    out = st.integrate(y, t)
    assert np.max(np.abs(out - np.sin(t))) < 1e-3


def test_integrate_constant_gives_ramp():
    t = np.linspace(0.0, 10.0, 5_000)
    y = np.full_like(t, 3.0)
    out = st.integrate(y, t)
    assert np.allclose(out, 3.0 * t, atol=1e-6)


def test_integrate_negate_flips_sign():
    t = np.linspace(0.0, 5.0, 2_000)
    y = np.sin(t)
    assert np.allclose(st.integrate(y, t, negate=True), -st.integrate(y, t))


def test_gradient_sin_gives_cos():
    # d/dt sin(t) = cos(t)
    t = np.linspace(0.0, 2 * np.pi, 20_000)
    g = st.gradient(np.sin(t), t)
    assert np.max(np.abs(_interior(g) - _interior(np.cos(t)))) < 1e-3


def test_gradient_then_integrate_roundtrip():
    # integrate(gradient(y)) recovers y up to its initial value (here 0)
    t = np.linspace(0.0, 3.0, 10_000)
    y = t ** 2                      # y(0) = 0
    recon = st.integrate(st.gradient(y, t), t)
    assert np.max(np.abs(_interior(recon) - _interior(y))) < 1e-2


def test_lowpass_passes_low_attenuates_high():
    # fs = 10 kHz; 5 Hz tone (well below cutoff) must pass ~unchanged,
    # 500 Hz tone (decade above cutoff) must be strongly attenuated.
    fs = 10_000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    low = np.sin(2 * np.pi * 5 * t)
    high = 0.5 * np.sin(2 * np.pi * 500 * t)
    filt = st.lowpass(low + high, t, cutoff_hz=50.0)
    # low component preserved (zero-phase filtfilt -> no shift to correct for)
    assert np.max(np.abs(_interior(filt) - _interior(low))) < 0.05
    # the high-frequency energy is gone: residual std collapses
    raw_high_std = np.std(_interior(high))
    resid_std = np.std(_interior(filt - low))
    assert resid_std < 0.15 * raw_high_std


def test_lowpass_short_input_is_noop():
    t = np.array([0.0, 1.0])
    y = np.array([1.0, -1.0])
    assert np.array_equal(st.lowpass(y, t, 10.0), y)


def test_movmean_reduces_noise_variance():
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, 10_000)
    sm = st.movmean(y, 50)
    assert np.var(_interior(sm)) < 0.1 * np.var(_interior(y))


def test_baseline_subtracts_pretrigger_mean():
    t = np.linspace(-1.0, 1.0, 4_000)
    y = np.where(t < 0, 5.0, 105.0)         # 5 before t=0, 105 after
    out = st.baseline(y, t, end=0.0)        # subtract mean over t<=0 (~5)
    pre = out[t < 0]
    assert abs(float(np.mean(pre))) < 1e-6  # pre-trigger now ~0


def test_combine_columns_ops_and_guards():
    a = np.array([2., 4, 6, 8]); b = np.array([1., 2, 3, 4])
    assert list(st.combine_columns(a, b, "+")) == [3, 6, 9, 12]
    assert list(st.combine_columns(a, b, "-")) == [1, 2, 3, 4]
    assert list(st.combine_columns(a, b, "*")) == [2, 8, 18, 32]
    assert list(st.combine_columns(a, b, "/")) == [2, 2, 2, 2]
    with pytest.raises(st.FormulaError):
        st.combine_columns(a, np.array([1., 0, 3, 4]), "/")     # div by zero
    with pytest.raises(st.FormulaError):
        st.combine_columns(a, np.array([1., 2, 3]), "+")        # length mismatch


def test_evaluate_formula_sibling_columns():
    t = np.linspace(0, 1, 4); a = np.array([2., 4, 6, 8]); b = np.array([1., 2, 3, 4])
    assert list(st.evaluate_formula("CH1/CH2", a, t, columns={"CH1": a, "CH2": b})) == [2, 2, 2, 2]
    out = st.evaluate_formula('col("CH a") - CH2', a, t, columns={"CH a": a, "CH2": b})
    assert list(out) == [1, 2, 3, 4]
