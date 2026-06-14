"""
Smoke + contract tests for Scope Studio's deterministic NumPy engine.

Coverage:
  * csv_loader   — Tektronix preamble/header/units detection; decimation
  * signal_tools — safe formula evaluation AND rejection of unsafe input
  * calibration  — forced-through-origin gain fit
  * detect_anomalies — spike detection on a known injected spike
  * plot_render  — display decimation threshold behavior (headless import)
  * model_catalog — profile lookup + fallback

Design-rule guard: every numeric assertion below is checked against a
value NumPy computed here, never a number the LLM was trusted to produce.
"""
from __future__ import annotations

import numpy as np
import pytest

import csv_loader
import signal_tools
import calibration
import detect_anomalies
import plot_render
import model_catalog


# ----------------------------------------------------------- csv_loader ----
def test_load_csv_header_units_and_shape(tek_csv):
    d = csv_loader.load_csv(str(tek_csv))
    assert d.columns == ["TIME", "CH1", "CH2"]
    assert d.n_rows == 200
    # Vertical Units row maps onto the channels after TIME
    assert d.units.get("CH1") == "V"
    assert d.units.get("CH2") == "A"
    # CH2 is a clean 0..1000 ramp
    assert d.df["CH2"].max() == pytest.approx(1000.0, rel=1e-3)


def test_minmax_decimate_preserves_peak():
    n = 100_000
    x = np.arange(n, dtype=np.float64)
    y = np.sin(x * 0.01)
    y[54321] = 999.0                       # a spike that must survive
    xd, yd = csv_loader.minmax_decimate(x, y, target=2000)
    assert len(xd) < n
    assert np.nanmax(yd) == pytest.approx(999.0, abs=1e-9)


def test_minmax_decimate_noop_when_small():
    x = np.arange(10, dtype=np.float64)
    y = x.copy()
    xd, yd = csv_loader.minmax_decimate(x, y, target=2000)
    assert len(xd) == 10            # below target -> returned unchanged


# --------------------------------------------------------- signal_tools ----
def test_formula_basic_arithmetic(ramp_axes):
    x, t_s = ramp_axes
    out = signal_tools.evaluate_formula("x * 2", x, t_s)
    assert np.allclose(out, x * 2)


def test_formula_demean_centers(ramp_axes):
    x, t_s = ramp_axes
    out = signal_tools.evaluate_formula("demean(x)", x, t_s)
    assert abs(float(np.mean(out))) < 1e-9


def test_formula_rejects_imports(ramp_axes):
    x, t_s = ramp_axes
    with pytest.raises(signal_tools.FormulaError):
        signal_tools.evaluate_formula("__import__('os').system('echo hi')",
                                      x, t_s)


def test_formula_rejects_attribute_access(ramp_axes):
    x, t_s = ramp_axes
    with pytest.raises(signal_tools.FormulaError):
        signal_tools.evaluate_formula("x.__class__", x, t_s)


def test_formula_rejects_scalar_result(ramp_axes):
    x, t_s = ramp_axes
    with pytest.raises(signal_tools.FormulaError):
        signal_tools.evaluate_formula("1.0", x, t_s)


# ----------------------------------------------------------- calibration ----
def test_forced_origin_gain_recovers_slope():
    x = np.linspace(0, 10, 500)
    y_src = np.linspace(0, 100, 500)
    y_ref = 2.5 * y_src                       # exact gain of 2.5
    res = calibration.fit_forced_origin_gain(x, y_src, y_ref, lo=0, hi=10)
    assert res.slope == pytest.approx(2.5, rel=1e-9)
    assert res.r2 == pytest.approx(1.0, abs=1e-9)
    assert res.n_samples == 500


def test_forced_origin_gain_raises_on_empty_window():
    x = np.linspace(0, 10, 500)
    y = np.linspace(0, 100, 500)
    with pytest.raises(calibration.CalibrationError):
        calibration.fit_forced_origin_gain(x, y, y, lo=100, hi=200)


# ------------------------------------------------------ detect_anomalies ----
def test_detect_finds_injected_spike():
    n = 20_000
    x = np.linspace(0, 50, n)                 # ms
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1.0, n)
    y[12_000] += 80.0                         # unmistakable spike
    rep = detect_anomalies.detect(x, {"CH2 (A)": y}, threshold_sigma=6.0)
    assert len(rep.findings) == 1
    assert rep.findings[0].n_spike_events >= 1


# ----------------------------------------------------------- plot_render ----
def test_decimate_for_render_noop_below_threshold():
    x = np.arange(1000, dtype=np.float64)
    y = x.copy()
    xd, yd = plot_render.decimate_for_render(x, y, threshold=5000)
    assert xd.size == 1000                    # untouched (display-only)


def test_decimate_for_render_reduces_above_threshold():
    n = 50_000
    x = np.arange(n, dtype=np.float64)
    y = np.sin(x * 0.001)
    xd, yd = plot_render.decimate_for_render(x, y, threshold=10_000,
                                             target=4000)
    assert xd.size < n


def test_enable_view_downsampling_contract():
    """Regression guard for the blank-plot bug: the view-dependent settings
    (auto peak-downsampling + clip-to-view) must be applied by
    enable_view_downsampling — which callers invoke AFTER addItem — not
    folded back into make_curve (which runs before the curve has a view).
    Uses a recording stub so no Qt is needed."""
    class FakeCurve:
        def __init__(self):
            self.calls = {}
        def setDownsampling(self, **kw):
            self.calls["downsampling"] = kw
        def setClipToView(self, val):
            self.calls["clip"] = val

    c = FakeCurve()
    plot_render.enable_view_downsampling(c)
    assert c.calls["downsampling"] == {"auto": True, "method": "peak"}
    assert c.calls["clip"] is True


# ---------------------------------------------------------- model_catalog ----
def test_model_catalog_lookup_and_fallback():
    names = model_catalog.profile_names()
    assert names, "expected at least one model profile"
    first = model_catalog.profile_by_name(names[0])
    assert first.name == names[0]
    # unknown name -> a defined fallback profile, never a crash
    fb = model_catalog.profile_by_name("does-not-exist")
    assert isinstance(fb, model_catalog.ModelProfile)


def test_default_profile_lite_vs_full():
    lite = model_catalog.default_profile(lite=True)
    full = model_catalog.default_profile(lite=False)
    assert lite.tier == "lightweight"
    assert full.tier == "balanced"
    # a chat model, not the tiny action-only router
    assert "router" not in lite.role.lower()
    assert "action" not in lite.role.lower()
    assert "1b" in lite.model.lower()       # the Llama 3.2 1B chat model


def test_is_lite_reads_env(monkeypatch):
    monkeypatch.setenv("SCOPE_STUDIO_LITE", "1")
    assert model_catalog.is_lite() is True
    assert model_catalog.default_profile().tier == "lightweight"
    monkeypatch.delenv("SCOPE_STUDIO_LITE", raising=False)
    assert model_catalog.is_lite() is False
    assert model_catalog.default_profile().tier == "balanced"
