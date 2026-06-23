"""
detect_anomalies.py - Deterministic anomaly detection for scope channels.

Design rule: the local LLM must never compute numbers. This module produces
all quantitative findings with plain NumPy; the model only *routes* to it
and *interprets* its output afterwards.

Detectors (per channel, over the visible window):
  * Spikes   - robust z-score (MAD) on the residual after a moving-mean,
               grouped into events, with a ringing-frequency estimate from
               zero crossings (e.g. the ~138 kHz EMP pickup seen on bus-bar
               monitors but not the Pearson).
  * Clipping - long runs pinned at the global extremes (scope saturation).
  * Drift    - linear slope over the first 10% of the window vs noise.
  * Crest    - peak/RMS ratio; >5 suggests narrow spikes dominate.
  * Dropouts - NaN samples.
Cross-channel:
  * Imbalance between current-like channels (peak spread vs mean), the
    S1...S4 module-balance check.

All thresholds are arguments with sensible defaults so the side chat can
tune them via actions, e.g. {"run": "detect_anomalies", "threshold_sigma": 5}.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ChannelFindings:
    label: str
    items: list[str] = field(default_factory=list)
    n_spike_events: int = 0


@dataclass
class AnomalyReport:
    window: str
    findings: list[ChannelFindings] = field(default_factory=list)
    cross: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [f"Anomaly scan over {self.window}:"]
        any_hit = False
        for f in self.findings:
            if not f.items:
                lines.append(f"- {f.label}: no anomalies detected")
                continue
            any_hit = True
            lines.append(f"- {f.label}:")
            lines.extend(f"    * {it}" for it in f.items)
        if self.cross:
            any_hit = True
            lines.append("- Cross-channel:")
            lines.extend(f"    * {it}" for it in self.cross)
        if not any_hit:
            lines.append("(clean shot by these criteria)")
        return "\n".join(lines)


def _moving_mean(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y.copy()
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def _robust_sigma(r: np.ndarray) -> float:
    med = np.nanmedian(r)
    mad = np.nanmedian(np.abs(r - med))
    return float(1.4826 * mad) if mad > 0 else float(np.nanstd(r)) or 1e-30


def _group_events(idx: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Group sorted indices into (start, end) events separated by > gap."""
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > gap)[0]
    starts = np.concatenate(([0], splits + 1))
    ends = np.concatenate((splits, [idx.size - 1]))
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def _ringing_freq(residual: np.ndarray, x: np.ndarray,
                  center: int, halfspan: int) -> float | None:
    """Estimate oscillation frequency around an event from zero crossings of
    the residual. Returns frequency in 1/(x units), or None."""
    a = max(0, center - halfspan)
    b = min(len(residual), center + halfspan)
    seg, xs = residual[a:b], x[a:b]
    if seg.size < 8:
        return None
    s = np.signbit(seg - np.nanmean(seg))
    crossings = np.where(s[1:] != s[:-1])[0]
    if crossings.size < 4:
        return None
    span = xs[crossings[-1]] - xs[crossings[0]]
    if span <= 0:
        return None
    # two crossings per period
    return float((crossings.size - 1) / 2.0 / span)


def _long_flat_runs(x: np.ndarray, y: np.ndarray,
                    min_samples: int) -> list[tuple[int, int]]:
    """Find long runs of identical finite values in contiguous samples."""
    finite = np.isfinite(x) & np.isfinite(y)
    idx = np.where(finite)[0]
    if idx.size < max(2, min_samples):
        return []
    yv = y[idx]
    yrange = float(np.nanmax(yv) - np.nanmin(yv)) if yv.size else 0.0
    tol = max(1e-12, 1e-9 * max(1.0, abs(yrange)))
    flat_pair = (np.abs(np.diff(yv)) <= tol) & (np.diff(idx) == 1)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < flat_pair.size:
        if not flat_pair[i]:
            i += 1
            continue
        start = i
        while i < flat_pair.size and flat_pair[i]:
            i += 1
        end = i
        if end - start + 1 >= min_samples:
            runs.append((int(idx[start]), int(idx[end])))
    return runs


def analyze_channel(label: str, x: np.ndarray, y: np.ndarray,
                    threshold_sigma: float = 6.0,
                    crest_limit: float = 5.0,
                    clip_run: int = 8,
                    max_events_reported: int = 6,
                    x_unit: str = "ms") -> ChannelFindings:
    f = ChannelFindings(label=label)
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = y.size
    if n < 16:
        f.items.append("window too small to analyze")
        return f

    raw_y = y.copy()
    nan_count = int(np.isnan(y).sum())
    if nan_count:
        f.items.append(f"{nan_count:,} NaN/dropout samples "
                       f"({100*nan_count/n:.2f}% of window)")
        y = np.nan_to_num(y, nan=float(np.nanmedian(y)))

    flat_runs = _long_flat_runs(x, raw_y, min_samples=max(clip_run, n // 300))
    if flat_runs:
        longest = max(flat_runs, key=lambda ev: ev[1] - ev[0] + 1)
        total = sum(b - a + 1 for a, b in flat_runs)
        a, b = longest
        value = float(np.nanmedian(raw_y[a:b + 1]))
        f.items.append(
            f"{len(flat_runs)} flatline/dropout candidate run(s); "
            f"{100*total/n:.2f}% of window flat. Longest: "
            f"t={x[a]:.4g}-{x[b]:.4g} {x_unit}, "
            f"{b-a+1:,} samples at {value:.4g}")

    # --- spikes via robust z on moving-mean residual --------------------
    w = max(5, n // 2000)
    residual = y - _moving_mean(y, w)
    sigma = _robust_sigma(residual)
    z = np.abs(residual) / sigma
    hits = np.where(z > threshold_sigma)[0]
    events = _group_events(hits, gap=w)
    f.n_spike_events = len(events)
    # periodic-event guard: a PWM/control square wave triggers the spike
    # detector at every edge. If many events occur with regular spacing,
    # report them as switching transitions, not anomalies.
    if len(events) >= 20:
        centers = np.array([x[(a + b) // 2] for a, b in events])
        gaps = np.diff(np.sort(centers))
        gaps = gaps[gaps > 0]
        if gaps.size >= 10:
            # periodic trains (even with asymmetric duty cycle) produce gaps
            # clustered on a few discrete values; random anomalies don't.
            med = float(np.median(gaps))
            bins = np.round(gaps / max(med * 0.10, 1e-12))
            uniq, counts = np.unique(bins, return_counts=True)
            top_cover = counts[np.argsort(counts)][-4:].sum() / gaps.size
            if top_cover >= 0.8:
                f.items.append(
                    f"{len(events)} periodic transition events, median "
                    f"spacing {med:.4g} {x_unit} "
                    f"(~{1.0/med:.4g} per {x_unit}) - consistent with "
                    "PWM/switching edges, not random anomalies")
                events = []
    if events:
        # strongest events first
        scored = sorted(events, key=lambda ev: -float(
            np.max(z[ev[0]:ev[1] + 1])))
        descr = []
        for (a, b) in scored[:max_events_reported]:
            k = a + int(np.argmax(np.abs(residual[a:b + 1])))
            amp = residual[k]
            freq = _ringing_freq(residual, x, k, halfspan=max(w * 4, 200))
            d = (f"t={x[k]:.4g} {x_unit}, amplitude {amp:+.4g} "
                 f"({z[k]:.1f}σ)")
            if freq:
                d += f", ~{freq:.3g} cycles/{x_unit} ringing"
            descr.append(d)
        f.items.append(
            f"{len(events)} spike event(s) above {threshold_sigma:g}σ; "
            "strongest: " + "; ".join(descr))

    # --- clipping / saturation ------------------------------------------
    ymax, ymin = float(np.max(y)), float(np.min(y))
    rng = ymax - ymin
    if rng > 0:
        for name, lvl in (("max", ymax), ("min", ymin)):
            pinned = np.abs(y - lvl) < 1e-9 * max(abs(lvl), rng)
            runs = _group_events(np.where(pinned)[0], gap=1)
            long_runs = [r for r in runs if r[1] - r[0] + 1 >= clip_run]
            pinned_frac = sum(r[1] - r[0] + 1 for r in long_runs) / n
            if long_runs and pinned_frac > 0.005:
                f.items.append(
                    f"possible clipping at {name} ({lvl:.4g}): "
                    f"{100*pinned_frac:.1f}% of window in flat runs "
                    f">= {clip_run} samples")

    # --- baseline drift over the first 10% ------------------------------
    head = slice(0, max(16, n // 10))
    xh, yh = x[head], y[head]
    if np.ptp(xh) > 0:
        slope = float(np.polyfit(xh, yh, 1)[0])
        drift = slope * np.ptp(xh)
        if abs(drift) > 5 * sigma and abs(drift) > 0.01 * rng:
            f.items.append(
                f"baseline drift in first 10% of window: "
                f"{drift:+.4g} over {np.ptp(xh):.4g} {x_unit} "
                f"(slope {slope:+.4g}/{x_unit})")

    # --- crest factor -----------------------------------------------------
    rms = float(np.sqrt(np.mean(y ** 2)))
    pk = float(np.max(np.abs(y)))
    if rms > 0 and pk / rms > crest_limit:
        f.items.append(
            f"crest factor {pk/rms:.1f} (peak {pk:.4g} vs RMS {rms:.4g}) - "
            "narrow spikes dominate; check if real or EMI pickup")
    return f


def detect(x: np.ndarray, channels: dict[str, np.ndarray],
           threshold_sigma: float = 6.0, crest_limit: float = 5.0,
           imbalance_limit: float = 0.10, x_unit: str = "ms",
           window_label: str = "") -> AnomalyReport:
    rep = AnomalyReport(window=window_label or
                        f"{x[0]:.4g}-{x[-1]:.4g} {x_unit}, "
                        f"{len(x):,} samples")
    for label, y in channels.items():
        rep.findings.append(analyze_channel(
            label, x, y, threshold_sigma=threshold_sigma,
            crest_limit=crest_limit, x_unit=x_unit))

    # cross-channel module balance among current-like channels
    current_like = {lab: y for lab, y in channels.items()
                    if "(a)" in lab.lower() or "current" in lab.lower()}
    if len(current_like) >= 2:
        peaks = {lab: float(np.nanmax(np.abs(y)))
                 for lab, y in current_like.items()}
        mean_pk = np.mean(list(peaks.values()))
        if mean_pk > 0:
            spread = {lab: (p - mean_pk) / mean_pk
                      for lab, p in peaks.items()}
            worst = max(spread.items(), key=lambda kv: abs(kv[1]))
            if abs(worst[1]) > imbalance_limit:
                detail = ", ".join(f"{lab}: {p:.4g}"
                                   for lab, p in peaks.items())
                rep.cross.append(
                    f"module imbalance: {worst[0]} deviates "
                    f"{100*worst[1]:+.1f}% from the mean peak "
                    f"({detail})")
    return rep
