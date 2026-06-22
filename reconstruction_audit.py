"""Deterministic reconstruction audit for censored current pulses.

The audit is intentionally not an AI interpretation layer. It runs the
existing numerical tools, compares their outputs, and explains whether the
RLC reconstruction is internally consistent enough to trust for a figure or
engineering decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np

from rlc_reconstruct import fit_rlc
from saturation_recovery import estimate_true_current


@dataclass
class AuditReport:
    ok: bool
    text: str
    overlay: dict | None = None
    params: dict[str, Any] = field(default_factory=dict)


def _finite_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _ratio_factor(value) -> float | None:
    value = _finite_float(value)
    if value is None or value <= 0:
        return None
    return max(value, 1.0 / value)


def _spread_pct(estimates: dict[str, float]) -> float | None:
    vals = np.asarray([abs(v) for v in estimates.values()
                       if np.isfinite(v) and abs(v) > 0], dtype=np.float64)
    if vals.size < 2:
        return None
    med = float(np.median(vals))
    if med <= 0:
        return None
    return 100.0 * float(np.nanmax(vals) - np.nanmin(vals)) / med


def _sat_estimates_from_report(text: str, overlay: dict | None) -> dict[str, float]:
    estimates: dict[str, float] = {}
    if overlay:
        clip = overlay.get("clip") or ()
        clip_start = _finite_float(clip[0]) if len(clip) >= 1 else None
        if clip_start is not None and isinstance(overlay.get("droop"), dict):
            d = overlay["droop"]
            a, b = _finite_float(d.get("a")), _finite_float(d.get("b"))
            if a is not None and b is not None:
                estimates["saturation droop projection"] = a * clip_start + b
        if isinstance(overlay.get("rise"), dict):
            r = overlay["rise"]
            a, b = _finite_float(r.get("a")), _finite_float(r.get("b"))
            if a is not None and b is not None and clip_start is not None:
                estimates["rise-side projection"] = a * clip_start + b
        inter = overlay.get("intersection")
        if isinstance(inter, (list, tuple)) and len(inter) >= 2:
            peak = _finite_float(inter[1])
            if peak is not None:
                estimates["two-slope intersection"] = peak

    # Optional cross-calibration estimate is currently text-only in
    # saturation_recovery; parsing it keeps the audit useful without changing
    # that tool's public dataclass shape.
    m = re.search(r"estimated true peak\s+([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                  text)
    if m:
        estimates["reference cross-calibration"] = float(m.group(1))
    return estimates


def _format_estimates(estimates: dict[str, float]) -> list[str]:
    if not estimates:
        return ["- no independent peak estimates available"]
    return [f"- {name}: {value:,.5g}" for name, value in estimates.items()]


def _format_ratio(name: str, ratio: float | None) -> tuple[str, float | None]:
    if ratio is None or not np.isfinite(ratio):
        return f"- {name}: not available", None
    factor = _ratio_factor(ratio)
    return f"- {name}: fit/expected = {ratio:.3g} ({factor:.3g}x mismatch)", factor


def _verdict(residual_pct: float | None, method_spread: float | None,
             physical_factor: float | None, rlc_ok: bool) -> tuple[str, str]:
    if not rlc_ok:
        return "DO NOT TRUST YET", "RLC fit failed or lacked enough trusted samples."
    if residual_pct is not None and residual_pct > 5.0:
        return "DO NOT TRUST YET", "Clean-window residual is too large."
    if method_spread is not None and method_spread > 30.0:
        return "DO NOT TRUST YET", "Peak-estimation methods strongly disagree."
    if physical_factor is not None and physical_factor > 4.0:
        return "DO NOT TRUST YET", "Fit is far from supplied physical R/L/C/V0 hints."
    if ((method_spread is not None and method_spread > 10.0)
            or (physical_factor is not None and physical_factor > 2.0)
            or (residual_pct is not None and residual_pct > 2.0)):
        return "PLAUSIBLE BUT MODEL MISMATCH", (
            "Overlay can be useful, but at least one consistency check is weak.")
    return "RELIABLE WITH CURRENT ASSUMPTIONS", (
        "Residuals, method agreement, and supplied physics are mutually consistent.")


def _recommendations(residual_pct: float | None, method_spread: float | None,
                     physical: dict[str, Any], sensitivity_spread: float | None) -> list[str]:
    recs: list[str] = []
    tau_r_factor = _ratio_factor(physical.get("tau_r_over_expected"))
    tau_d_factor = _ratio_factor(physical.get("tau_d_over_expected"))
    slope_factor = _ratio_factor(physical.get("initial_slope_over_expected"))
    if residual_pct is not None and residual_pct > 2.0:
        recs.append("clean-window residual is high: narrow trusted windows or add missing dynamics.")
    if method_spread is not None and method_spread > 10.0:
        recs.append("peak methods disagree: re-check saturation level, reference channel, and calibration window.")
    if tau_r_factor is not None and tau_r_factor > 1.5:
        recs.append("tau_rise differs from L/R: verify effective inductance, resistance, onset time, and filter cutoff.")
    if tau_d_factor is not None and tau_d_factor > 1.5:
        recs.append("tau_droop differs from R*C: check usable capacitance, ESR/contact resistance, snubber/switch paths, or active control.")
    if slope_factor is not None and slope_factor > 1.5:
        recs.append("initial slope differs from V0/L: check charge voltage, inductance, polarity, and early-time trusted window.")
    if sensitivity_spread is not None and sensitivity_spread > 10.0:
        recs.append("peak is sensitive to R/L/C/V0: report a sensitivity band before using the number in a claim.")
    if not recs:
        recs.append("document the assumptions and keep the overlay labeled as a model estimate, not measured data.")
    return recs


def audit_reconstruction(t: np.ndarray, y: np.ndarray, *,
                         label: str = "",
                         sat_level: float | None = None,
                         y_ref: np.ndarray | None = None,
                         ref_label: str = "",
                         t_window: tuple[float, float] | None = None,
                         ref_window: tuple[float, float] | None = None,
                         trusted_windows: list[tuple[float, float]] | None = None,
                         resistance_ohm: float | None = None,
                         inductance_h: float | None = None,
                         capacitance_f: float | None = None,
                         charging_voltage_v: float | None = None,
                         physical_prior_weight: float = 0.0,
                         sensitivity_pct: float = 0.10,
                         run_sensitivity: bool = True) -> AuditReport:
    """Run saturation + RLC reconstruction and score consistency.

    All inputs are display/session arrays. The caller remains responsible for
    read-only CSV policy; this function only returns text, parameters, and an
    overlay curve.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.size < 256 or y.size < 256:
        return AuditReport(False, "Reconstruction audit: not enough samples.")

    sat = estimate_true_current(
        t, y, label=label, y_ref=y_ref, ref_label=ref_label,
        cal_window=ref_window, sat_level=sat_level)
    physical_kwargs = {
        "resistance_ohm": resistance_ohm,
        "inductance_h": inductance_h,
        "capacitance_f": capacitance_f,
        "charging_voltage_v": charging_voltage_v,
        "physical_prior_weight": physical_prior_weight,
    }
    rlc = fit_rlc(t, y, sat_level=sat_level, label=label,
                  t_window=t_window, y_ref=y_ref, ref_window=ref_window,
                  ref_label=ref_label, trusted_windows=trusted_windows,
                  **physical_kwargs)
    if not rlc.ok:
        return AuditReport(False, "Reconstruction audit failed:\n\n" + rlc.text,
                           params={"rlc": rlc.params, "saturation": sat.text})

    estimates = _sat_estimates_from_report(sat.text, sat.overlay)
    rlc_peak = _finite_float(rlc.params.get("peak"))
    if rlc_peak is not None:
        estimates["censored RLC"] = rlc_peak
    method_spread = _spread_pct(estimates)

    peak = abs(float(rlc.params.get("peak", 0.0) or 0.0))
    rms = abs(float(rlc.params.get("rms", np.nan)))
    residual_pct = None if not peak else 100.0 * rms / peak
    physical = dict(rlc.params.get("physical") or {})
    ratio_lines: list[str] = []
    factors = []
    for label_ratio, key in (
        ("tau_rise vs L/R", "tau_r_over_expected"),
        ("tau_droop vs R*C", "tau_d_over_expected"),
        ("initial slope vs V0/L", "initial_slope_over_expected"),
    ):
        line, factor = _format_ratio(label_ratio, physical.get(key))
        ratio_lines.append(line)
        if factor is not None:
            factors.append(factor)
    physical_factor = max(factors) if factors else None

    sensitivity: dict[str, Any] = {"ran": False}
    if run_sensitivity and physical_prior_weight > 0 and sensitivity_pct > 0:
        variants: dict[str, float] = {}
        base_kwargs = dict(physical_kwargs)
        for key in ("resistance_ohm", "inductance_h",
                    "capacitance_f", "charging_voltage_v"):
            value = _finite_float(base_kwargs.get(key))
            if value is None or value == 0:
                continue
            for sign, tag in ((-1.0, "-"), (1.0, "+")):
                trial = dict(base_kwargs)
                trial[key] = value * (1.0 + sign * sensitivity_pct)
                rep = fit_rlc(t, y, sat_level=sat_level, label=label,
                              t_window=t_window, y_ref=y_ref,
                              ref_window=ref_window, ref_label=ref_label,
                              trusted_windows=trusted_windows,
                              n_boot=0, **trial)
                if rep.ok and _finite_float(rep.params.get("peak")) is not None:
                    variants[f"{key} {tag}{100*sensitivity_pct:.0f}%"] = float(rep.params["peak"])
        sensitivity = {
            "ran": bool(variants),
            "pct": sensitivity_pct,
            "peaks": variants,
            "spread_pct": _spread_pct(variants),
        }
    sensitivity_spread = sensitivity.get("spread_pct") if sensitivity.get("ran") else None

    verdict, verdict_reason = _verdict(
        residual_pct, method_spread, physical_factor, rlc.ok)
    recs = _recommendations(residual_pct, method_spread, physical,
                            sensitivity_spread)

    lines = [
        "Reconstruction audit",
        "====================",
        f"Verdict: {verdict}",
        f"Reason: {verdict_reason}",
        "",
        "Peak estimates",
        *(_format_estimates(estimates)),
    ]
    if method_spread is not None:
        lines.append(f"- method spread: {method_spread:.2f}% of median peak")
    lines += [
        "",
        "Fit quality",
        f"- clean-window RMS residual: {rms:.4g} A"
        + (f" ({residual_pct:.2f}% of reconstructed peak)"
           if residual_pct is not None else ""),
        f"- clean samples: {rlc.params.get('n_clean', 'n/a')}",
        f"- censored/lower-bound samples: {rlc.params.get('n_censored', 'n/a')}",
        f"- reference samples: {rlc.params.get('n_reference', 'n/a')}",
        "",
        "Physical consistency",
        *ratio_lines,
    ]
    if sensitivity.get("ran"):
        lines += [
            "",
            f"Sensitivity sweep (+/- {100*sensitivity_pct:.0f}% physical hints)",
            *[f"- {name}: peak {value:,.5g}" for name, value
              in sensitivity["peaks"].items()],
        ]
        if sensitivity_spread is not None:
            lines.append(f"- sensitivity peak spread: {sensitivity_spread:.2f}%")
    elif physical_prior_weight <= 0:
        lines += [
            "",
            "Sensitivity sweep",
            "- skipped because physical prior weight is 0; R/L/C/V0 are report-only.",
        ]
    lines += [
        "",
        "Recommended next checks",
        *[f"- {r}" for r in recs],
        "",
        "--- Saturation estimate details ---",
        sat.text,
        "",
        "--- RLC reconstruction details ---",
        rlc.text,
    ]
    return AuditReport(
        True, "\n".join(lines), overlay=rlc.curve,
        params={
            "verdict": verdict,
            "residual_pct": residual_pct,
            "method_spread_pct": method_spread,
            "physical_mismatch_factor": physical_factor,
            "estimates": estimates,
            "sensitivity": sensitivity,
            "rlc": rlc.params,
            "saturation_text": sat.text,
        })
