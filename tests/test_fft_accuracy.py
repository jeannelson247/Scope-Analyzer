"""Closed-form accuracy tests for the Detail+FFT amplitude spectrum.

These guard the only math claim in the Detail/ringing view (used by
surface3d): a known tone must be recovered at the right frequency and
amplitude. Pure NumPy, headless.
"""
from __future__ import annotations

import numpy as np

import signal_tools as st


def _tone(freq_hz, amp, fs, n, phase=0.0):
    t = np.arange(n) / fs
    return t, amp * np.sin(2 * np.pi * freq_hz * t + phase)


def test_amplitude_spectrum_recovers_bin_centered_tone():
    # fs=2 MHz, n=4000 -> bin width 500 Hz; 138 kHz = bin 276 exactly,
    # so coherent-gain normalization recovers the amplitude with no
    # scalloping loss.
    fs, n = 2_000_000.0, 4000
    f0, A = 138_000.0, 5.0
    assert abs((f0 / (fs / n)) - round(f0 / (fs / n))) < 1e-9   # bin-centered
    _, y = _tone(f0, A, fs, n)
    freq, amp = st.amplitude_spectrum(y, dt=1.0 / fs)
    k = int(np.argmax(amp))
    assert abs(freq[k] - f0) < (fs / n)              # within one bin
    assert np.isclose(amp[k], A, rtol=0.05)          # amplitude ~A (±5%)


def test_dominant_frequency_ignores_dc_and_low_band():
    # A large DC offset + low drift must not beat the real tone above f_min.
    fs, n = 1_000_000.0, 8000
    t, y = _tone(45_000.0, 3.0, fs, n)
    y = y + 100.0 + 2.0 * t                          # DC + slow ramp
    f = st.dominant_frequency(y, dt=1.0 / fs, f_min=1_000.0)
    assert abs(f - 45_000.0) < 2 * (fs / n)


def test_amplitude_spectrum_degenerate_inputs():
    f, a = st.amplitude_spectrum(np.array([1.0]), dt=1e-6)   # too short
    assert f.size == 0 and a.size == 0
    f2, a2 = st.amplitude_spectrum(np.ones(64), dt=0.0)      # bad dt
    assert f2.size == 0 and a2.size == 0
