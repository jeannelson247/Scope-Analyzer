"""
rlc_reconstruct.py - Gray-box reconstruction of a saturated current pulse.

Model: the overdamped series-RLC discharge solution

    I(t) = A * ( exp(-(t-t0)/tau_d) - exp(-(t-t0)/tau_r) ),  t > t0

with 4 parameters (A, t0, tau_r ~ L/R, tau_d ~ R*C). Fitted by censored
maximum likelihood (Tobit-style):

  * clean samples (reading below the saturation level, outside the
    censored hull) -> ordinary least-squares residuals;
  * censored samples (sensor saturated; reading is only a LOWER bound)
    -> one-sided hinge penalty when the model falls BELOW the level.
    The gap actively constrains the fit instead of being discarded.

Uncertainty: residual-bootstrap refits -> pointwise 95% band over the
window and a CI on the reconstructed peak.

Outputs both a text report and overlay curves (median + band) so the
reconstruction can be drawn over the measured traces for visual
comparison. Deterministic NumPy/SciPy only - the LLM never computes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

N_FIT = 6000          # decimated sample budget for the fit
N_BOOT = 25           # bootstrap refits
N_CURVE = 400         # overlay curve resolution
CENSOR_WEIGHT = 3.0   # hinge weight for censored samples


@dataclass
class RLCReport:
    label: str
    ok: bool
    text: str
    # overlay: {"t": [...], "y": [...], "lo": [...], "hi": [...]}
    curve: dict | None = None
    params: dict = field(default_factory=dict)


def rlc_model(t: np.ndarray, A: float, t0: float,
              tau_r: float, tau_d: float) -> np.ndarray:
    out = np.zeros_like(t, dtype=np.float64)
    m = t > t0
    dt = t[m] - t0
    out[m] = A * (np.exp(-dt / tau_d) - np.exp(-dt / tau_r))
    return out


def _peak_of(A, t0, tau_r, tau_d):
    tp = t0 + (tau_r * tau_d / (tau_d - tau_r)) * np.log(tau_d / tau_r)
    return tp, float(rlc_model(np.array([tp]), A, t0, tau_r, tau_d)[0])


def _pack(A, t0, tau_r, tau_d):
    # positivity + tau_d > tau_r by construction
    return np.array([np.log(A), t0, np.log(tau_r), np.log(tau_d - tau_r)])


def _unpack(p):
    A = float(np.exp(p[0])); t0 = float(p[1])
    tau_r = float(np.exp(p[2])); tau_d = tau_r + float(np.exp(p[3]))
    return A, t0, tau_r, tau_d


def fit_rlc(t: np.ndarray, y: np.ndarray, sat_level: float | None = None,
            label: str = "",
            t_window: tuple[float, float] | None = None,
            y_ref: np.ndarray | None = None,
            ref_window: tuple[float, float] | None = None,
            ref_label: str = "",
            trusted_windows: list[tuple[float, float]] | None = None
            ) -> RLCReport:
    """t in display units (ms recommended), y the (possibly saturated)
    current channel, sat_level the known censoring threshold (None ->
    every sample treated as clean).

    t_window: fit ONLY this time range. Essential for switch-terminated
    pulses: the RLC model describes the free discharge, so the window
    must end before switch-off.
    trusted_windows: optional time ranges where the target monitor is known
    to be accurate. Outside these windows, non-censored target samples are
    excluded from the ordinary residuals while saturated samples still act as
    lower-bound constraints.
    y_ref / ref_window: a second sensor measuring the SAME current
    (e.g. a Pearson, trustworthy only before its core saturates) - its
    samples inside ref_window join the fit as additional clean data, so
    the reconstruction is consistent with both monitors."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y_ref is not None:
        y_ref = np.asarray(y_ref, dtype=np.float64)
    if t_window is not None:
        mwin = (t >= t_window[0]) & (t <= t_window[1])
        if mwin.sum() < 256:
            return RLCReport(label, False,
                             f"{label}: fit window {t_window} contains "
                             f"too few samples.")
        t, y = t[mwin], y[mwin]
        if y_ref is not None:
            y_ref = y_ref[mwin]
    # sign-normalize so the pulse is positive
    sgn = 1.0 if abs(np.nanmax(y)) >= abs(np.nanmin(y)) else -1.0
    ys = sgn * y

    # censored region: hull of samples above the level (cf.
    # saturation_recovery rationale), with a 2% guard band
    censored = np.zeros(len(ys), dtype=bool)
    if sat_level is not None:
        above = np.flatnonzero(ys >= sat_level)
        if above.size:
            a0, b0 = above[0], above[-1] + 1
            guard = 0.02 * abs(sat_level)
            while a0 > 0 and ys[a0 - 1] > sat_level - guard:
                a0 -= 1
            while b0 < len(ys) and ys[b0] > sat_level - guard:
                b0 += 1
            censored[a0:b0] = True

    # decimate for fitting speed (keep all censored-boundary detail)
    step = max(1, len(t) // N_FIT)
    tf, yf, cf = t[::step], ys[::step], censored[::step]
    if trusted_windows:
        trusted = np.zeros(len(tf), dtype=bool)
        for lo, hi in trusted_windows:
            a, b = sorted((float(lo), float(hi)))
            trusted |= (tf >= a) & (tf <= b)
    else:
        trusted = np.ones(len(tf), dtype=bool)
    clean = (~cf) & trusted
    if clean.sum() < 100:
        return RLCReport(label, False,
                         f"{label}: not enough trusted clean samples to fit.")
    # reference-sensor samples (same current, valid inside ref_window)
    if y_ref is not None and ref_window is not None:
        yrf = (sgn * y_ref)[::step]
        mref = (tf >= ref_window[0]) & (tf <= ref_window[1]) & \
            np.isfinite(yrf)
    else:
        yrf, mref = None, None

    # ---- initial guesses ------------------------------------------------
    pk = float(np.nanmax(yf[clean])) if sat_level is None else \
        max(float(np.nanmax(yf[clean])), sat_level) * 1.15
    rise_idx = np.flatnonzero(yf >= 0.1 * pk)
    t0_init = float(tf[rise_idx[0]]) if rise_idx.size else float(tf[0])
    span = float(tf[-1] - t0_init)
    tau_r_init = max(span * 0.02, 1e-6)
    tau_d_init = max(span * 1.0, tau_r_init * 10)
    scale = max(pk, 1e-9)

    def residuals(p):
        A, t0, tau_r, tau_d = _unpack(p)
        m = rlc_model(tf, A, t0, tau_r, tau_d)
        parts = [(yf[clean] - m[clean]) / scale]
        if mref is not None and mref.any():
            # second sensor's valid window: same current, equal weight
            parts.append((yrf[mref] - m[mref]) / scale)
        if cf.any() and sat_level is not None:
            # one-sided: penalize the model only for dipping BELOW the
            # censoring level inside the gap (truth is >= level there)
            parts.append(CENSOR_WEIGHT * np.maximum(
                0.0, sat_level - m[cf]) / scale)
        return np.concatenate(parts)

    p0 = _pack(pk * 1.2, t0_init - tau_r_init, tau_r_init, tau_d_init)
    sol = least_squares(residuals, p0, method="lm", max_nfev=2000) \
        if not cf.any() else least_squares(residuals, p0, max_nfev=2000)
    A, t0, tau_r, tau_d = _unpack(sol.x)
    tp, ip = _peak_of(A, t0, tau_r, tau_d)
    model_clean = rlc_model(tf, A, t0, tau_r, tau_d)
    rms = float(np.sqrt(np.mean((yf[clean] - model_clean[clean]) ** 2)))

    # ---- bootstrap ------------------------------------------------------
    rng = np.random.default_rng(0)
    res = yf[clean] - model_clean[clean]
    grid = np.linspace(float(tf[0]), float(tf[-1]), N_CURVE)
    # seed the pools with the point estimate so the band/CI always
    # contain the reported curve
    curves = [rlc_model(grid, A, t0, tau_r, tau_d)]
    peaks = [ip]
    for _ in range(N_BOOT):
        yb = yf.copy()
        yb[clean] = model_clean[clean] + rng.choice(res, clean.sum(),
                                                    replace=True)
        def res_b(p, yb=yb):
            Ab, t0b, trb, tdb = _unpack(p)
            mb = rlc_model(tf, Ab, t0b, trb, tdb)
            rb = [(yb[clean] - mb[clean]) / scale]
            if mref is not None and mref.any():
                rb.append((yrf[mref] - mb[mref]) / scale)
            if cf.any() and sat_level is not None:
                rb.append(CENSOR_WEIGHT * np.maximum(
                    0.0, sat_level - mb[cf]) / scale)
            return np.concatenate(rb)
        try:
            sb = least_squares(res_b, sol.x, max_nfev=500)
            Ab, t0b, trb, tdb = _unpack(sb.x)
            curves.append(rlc_model(grid, Ab, t0b, trb, tdb))
            peaks.append(_peak_of(Ab, t0b, trb, tdb)[1])
        except Exception:
            continue
    mid = rlc_model(grid, A, t0, tau_r, tau_d)
    if curves:
        stack = np.vstack(curves)
        # clamp band/CI to include the point estimate (percentile
        # interpolation can otherwise exclude it at small N_BOOT)
        lo = np.minimum(np.percentile(stack, 2.5, axis=0), mid)
        hi = np.maximum(np.percentile(stack, 97.5, axis=0), mid)
        ip_lo = min(float(np.percentile(peaks, 2.5)), ip)
        ip_hi = max(float(np.percentile(peaks, 97.5)), ip)
    else:
        lo = hi = mid
        ip_lo = ip_hi = ip
    n_cens = int(censored.sum())
    n_ref = int(mref.sum()) if mref is not None else 0
    trusted_note = ""
    if trusted_windows:
        spans = ", ".join(f"{float(a):.4g}..{float(b):.4g}"
                          for a, b in trusted_windows)
        trusted_note = f"  trusted target windows: {spans}\n"
    lines = [
        f"{label}: censored-ML RLC reconstruction "
        f"({clean.sum():,} clean fit samples"
        + (f", {n_cens:,} censored samples as lower bounds"
           if n_cens else ", no censoring")
        + (f", {n_ref:,} {ref_label or 'reference'} samples in "
           f"{ref_window}" if n_ref else "") + ")",
        trusted_note.rstrip(),
        f"  fitted tau_rise = {tau_r:.3g}, tau_droop = {tau_d:.4g} "
        f"(display units; compare to L/R and R*C of the rig)",
        f"  reconstructed peak I = {sgn*ip:,.5g} at t = {tp:.3g} "
        f"(95% bootstrap CI {sgn*ip_lo:,.5g} .. {sgn*ip_hi:,.5g})",
        f"  RMS residual on clean data: {rms:.4g} "
        f"({100*rms/max(ip,1e-12):.2f}% of peak)",
        "  note: validity rests on the overdamped-RLC assumption; if the "
        "overlaid curve visibly departs from clean data, the circuit "
        "model is missing dynamics (snubber, switch, core effects).",
    ]
    lines = [line for line in lines if line]
    return RLCReport(
        label, True, "\n".join(lines),
        curve={"t": grid.tolist(), "y": (sgn * mid).tolist(),
               "lo": (sgn * lo).tolist(), "hi": (sgn * hi).tolist()},
        params={"A": A, "t0": t0, "tau_r": tau_r, "tau_d": tau_d,
                "peak": sgn * ip, "peak_lo": sgn * ip_lo,
                "peak_hi": sgn * ip_hi, "rms": rms})
