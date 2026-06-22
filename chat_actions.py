"""
chat_actions.py - JSON action layer for the local AI side chat.

The model can end a reply with one fenced block:

```json
{"actions": [ ... ]}
```

Two kinds of actions:

1. FORMATTING - the model "types into" the same manual boxes the user has,
   so everything stays visible and hand-overridable afterwards:
     {"set_title": "..."} {"set_xlabel": "..."}
     {"set_ylabel_left": "..."} {"set_ylabel_right": "..."}
     {"set_xrange": {"min": -5, "max": 160}}
     {"set_yrange_left": {"min": -500, "max": 7000}}
     {"set_yrange_right": {"min": -5, "max": 20}}
     {"set_xscale": 1000} {"align_zeros": true}
     {"lowpass_filter": {"enabled": true, "cutoff_hz": 10000,
                         "target": "all|left|current"}}
     {"top_axis": {"on": true, "scale": 1.0, "label": "..."}}
     {"channel": {"name": "CH1", "enabled": true, "axis": "left|right",
                  "gain": 1.0, "offset": 0.0, "label": "...",
                  "formula": "(x-2.5)*1500/2"}}

2. TOOLS - deterministic scripts; the model routes, NumPy computes:
     {"run": "compute_stats"}
     {"run": "detect_anomalies", "threshold_sigma": 6,
      "crest_limit": 5, "imbalance_limit": 0.1}

`process_reply(win, reply)` returns:
    clean_text   - reply with the json block stripped
    applied      - human-readable list of formatting changes made
    tool_msgs    - output text of each tool that ran (post these to the chat
                   AND the history, so the model can interpret them next turn)
"""
from __future__ import annotations

import json
import re

import numpy as np

ACTION_SCHEMA = (
    "\n\nPLOT/TOOL ACTIONS: you may end your reply with ONE fenced block "
    "```json {\"actions\": [...]} ``` to change plot formatting or run a "
    "tool. Formatting actions: set_title, set_xlabel, set_ylabel_left, "
    "set_ylabel_right, set_xrange {min,max}, set_yrange_left {min,max}, "
    "set_yrange_right {min,max}, set_xscale <number>, align_zeros <bool>, "
    "lowpass_filter {enabled, cutoff_hz, target: all|left|current}, "
    "top_axis {on,scale,label}, channel {name, enabled, axis: left|right, "
    "gain, offset, label, formula}. Tool actions: {\"run\": "
    "\"compute_stats\"} and {\"run\": \"detect_anomalies\", "
    "\"threshold_sigma\": 6}, {\"run\": \"estimate_saturation\", "
    "\"sat_level\": 6000}, or {\"run\": \"reconstruct_rlc\", "
    "\"sat_level\": 6000, \"t_start\": 0, \"t_end\": 150, "
    "\"ref_end\": 5, \"resistance_ohm\": 0.068, "
    "\"inductance_uh\": 160, \"capacitance_f\": 2.24, "
    "\"charging_voltage_v\": 450}. Use {\"run\": \"rlc_audit\"} "
    "with the same reconstruction fields when the user asks whether a "
    "reconstruction is trustworthy. Ranges are in display units (x usually ms). "
    "Emit the block ONLY when the user asks to reformat, recalibrate, or "
    "scan/analyze for anomalies; otherwise reply normally without it. "
    "Never invent calibration numbers - use values from the user or the "
    "context. When a tool result appears in the conversation as 'tool:', "
    "interpret those numbers; do not recompute them."
)

_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


def extract_actions(reply: str) -> tuple[str, list[dict]]:
    m = _JSON_BLOCK.search(reply)
    if not m:
        return reply.strip(), []
    try:
        payload = json.loads(m.group(1))
        actions = payload.get("actions", [])
        if not isinstance(actions, list):
            actions = []
    except json.JSONDecodeError:
        return reply.strip(), []
    clean = (reply[:m.start()] + reply[m.end():]).strip()
    return clean, actions


# ------------------------------------------------------------------ tools --
def _visible_arrays(win) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Visible-window x and converted y for every enabled channel."""
    if getattr(win, "data", None) is None:
        return np.array([], dtype=np.float64), {}
    x = win._x()
    if x is None:
        return np.array([], dtype=np.float64), {}
    (x0, x1) = win.pi.vb.viewRange()[0]
    m = (x >= x0) & (x <= x1)
    xv = np.asarray(x[m], dtype=np.float64)
    chans: dict[str, np.ndarray] = {}
    for ch in win.channels:
        if not ch.enabled:
            continue
        try:
            chans[ch.display_label()] = np.asarray(
                win._channel_data(ch)[m], dtype=np.float64)
        except Exception:
            continue
    return xv, chans


def _parse_windows(value) -> list[tuple[float, float]] | None:
    """Parse optional trusted/clean time windows from JSON actions."""
    if value in (None, "", []):
        return None
    windows = []
    if isinstance(value, str):
        chunks = [p.strip() for p in re.split(r"[;,]", value) if p.strip()]
        for part in chunks:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", part)
            if len(nums) < 2:
                continue
            windows.append((float(nums[0]), float(nums[1])))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                lo = item.get("start", item.get("lo", item.get("min")))
                hi = item.get("end", item.get("hi", item.get("max")))
                if lo is not None and hi is not None:
                    windows.append((float(lo), float(hi)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                windows.append((float(item[0]), float(item[1])))
    return windows or None


def _optional_float(act: dict, *names: str) -> float | None:
    """Read optional numeric aliases from an LLM/tool action."""
    for name in names:
        value = act.get(name)
        if value in (None, "", "auto"):
            continue
        out = float(value)
        if np.isfinite(out):
            return out
    return None


def _physical_rlc_kwargs(act: dict) -> dict[str, float | None]:
    """Collect optional physical RLC hints for fit_rlc().

    These are deterministic tool parameters, not LLM-computed values. The
    model may route user-provided numbers into this dict, but NumPy/SciPy do
    the actual reconstruction.
    """
    L = _optional_float(act, "inductance_h", "l_h")
    if L is None:
        L_uh = _optional_float(
            act, "inductance_uh", "inductance_uH", "l_uh", "L_uH")
        L = None if L_uh is None else L_uh * 1e-6
    C = _optional_float(act, "capacitance_f", "c_f")
    if C is None:
        C_mf = _optional_float(act, "capacitance_mf", "c_mf")
        C = None if C_mf is None else C_mf * 1e-3
    prior = _optional_float(act, "physical_prior_weight", "prior_weight")
    return {
        "resistance_ohm": _optional_float(act, "resistance_ohm", "r_ohm"),
        "inductance_h": L,
        "capacitance_f": C,
        "charging_voltage_v": _optional_float(
            act, "charging_voltage_v", "initial_voltage_v", "v0_v",
            "capacitor_voltage_v"),
        "physical_prior_weight": 0.0 if prior is None else max(prior, 0.0),
    }


def run_tool(win, act: dict) -> str:
    name = str(act.get("run", ""))
    if name in ("compute_stats", "channel_stats"):   # router alias
        out = win.compute_stats()
        return out or "No stats available - load a file first."
    if name == "detect_anomalies":
        import detect_anomalies as da
        xv, chans = _visible_arrays(win)
        if xv.size < 16 or not chans:
            return "Anomaly scan: not enough visible data."
        rep = da.detect(
            xv, chans,
            threshold_sigma=float(act.get("threshold_sigma", 6.0)),
            crest_limit=float(act.get("crest_limit", 5.0)),
            imbalance_limit=float(act.get("imbalance_limit", 0.10)),
            x_unit=_x_unit(win))
        return rep.text()
    if name == "estimate_saturation":
        from saturation_recovery import estimate_true_current
        xv, chans = _visible_arrays(win)
        if xv.size < 64 or not chans:
            return "Saturation estimate: not enough visible data."
        want = str(act.get("channel", "")).lower()
        labels = list(chans)
        target = next((l for l in labels if want and want in l.lower()),
                      labels[0])
        ref_want = str(act.get("ref_channel",
                               act.get("reference", ""))).lower()
        refs = [l for l in labels if l != target]
        ref = next((l for l in refs if ref_want and ref_want in l.lower()),
                   None)
        if ref is None:
            ref = next(iter(refs), None)
        cal = None
        if "cal_start" in act and "cal_end" in act:
            cal = (float(act["cal_start"]), float(act["cal_end"]))
        sat = act.get("sat_level")
        rep = estimate_true_current(
            xv, chans[target], label=target,
            y_ref=chans[ref] if ref else None, ref_label=ref or "",
            cal_window=cal,
            sat_level=float(sat) if sat is not None else None)
        # hand the fit lines to the plot overlay (toggleable in the UI)
        if hasattr(win, "apply_sat_overlay"):
            if hasattr(win, "push_display_undo"):
                win.push_display_undo("AI/tool saturation overlay")
            win._sat_overlay = rep.overlay
            win.apply_sat_overlay()
        return rep.text
    if name == "zero_baseline":
        # set each enabled channel's offset so its pre-trigger (t < 0)
        # mean is exactly 0 - both monitors then start at the same level
        if getattr(win, "data", None) is None:
            return "Zero baseline: load a file first."
        if hasattr(win, "push_display_undo"):
            win.push_display_undo("AI/tool zero baseline")
        x = win._raw_x()
        out = []
        for r, ch in enumerate(win.channels):
            if not ch.enabled:
                continue
            try:
                y = win._channel_data(ch)
            except Exception:
                continue
            m = x < 0
            if m.sum() < 16:                  # no pre-trigger: first 5%
                m = np.zeros(len(y), dtype=bool)
                m[: max(16, len(y) // 20)] = True
            off = float(np.nanmean(y[m]))
            ch.offset = float(ch.offset) - off
            try:                              # keep the table in sync
                win.table.item(r, 5).setText(f"{ch.offset:g}")
            except Exception:
                pass
            out.append(f"{ch.display_label()}: removed {off:+.4g}")
        if not out:
            return "Zero baseline: no enabled channels."
        win._transform_cache.clear()
        win.refresh_plot()
        return ("Baselines zeroed on the pre-trigger window "
                "(offsets updated in the channel table):\n  "
                + "\n  ".join(out))
    if name in ("reconstruct_rlc", "rlc_audit", "audit_reconstruction"):
        from rlc_reconstruct import fit_rlc
        from reconstruction_audit import audit_reconstruction
        xv, chans = _visible_arrays(win)
        if xv.size < 256 or not chans:
            return "RLC reconstruction/audit: not enough visible data."
        want = str(act.get("channel", "")).lower()
        labels = list(chans)
        target = next((l for l in labels if want and want in l.lower()),
                      labels[0])
        sat = act.get("sat_level")
        # fit window (defaults to the visible range); must exclude
        # switch-off for switch-terminated pulses
        t_window = None
        if "t_start" in act or "t_end" in act:
            t_window = (float(act.get("t_start", xv[0])),
                        float(act.get("t_end", xv[-1])))
        # reference sensor (e.g. Pearson) valid up to ref_end
        ref_want = str(act.get("ref_channel",
                               act.get("reference", ""))).lower()
        refs = [l for l in labels if l != target]
        ref = next((l for l in refs if ref_want and ref_want in l.lower()),
                   None)
        if ref is None:
            ref = next(iter(refs), None)
        ref_arr, ref_window = None, None
        if ref is not None and "ref_end" in act:
            ref_arr = chans[ref]
            ref_window = (float(act.get("ref_start", xv[0])),
                          float(act["ref_end"]))
        trusted_windows = _parse_windows(
            act.get("trusted_windows", act.get("clean_windows")))
        physical = _physical_rlc_kwargs(act)
        if name == "reconstruct_rlc":
            rep = fit_rlc(xv, chans[target], label=target,
                          sat_level=float(sat) if sat is not None else None,
                          t_window=t_window, y_ref=ref_arr,
                          ref_window=ref_window, ref_label=ref or "",
                          trusted_windows=trusted_windows,
                          **physical)
            overlay = rep.curve
        else:
            sens = act.get("sensitivity_pct", 0.10)
            sens = float(sens)
            if sens > 1.0:
                sens *= 0.01
            run_sens = act.get("run_sensitivity", True)
            if isinstance(run_sens, str):
                run_sens = run_sens.strip().lower() not in (
                    "0", "false", "no", "off")
            rep = audit_reconstruction(
                xv, chans[target], label=target,
                sat_level=float(sat) if sat is not None else None,
                t_window=t_window, y_ref=ref_arr, ref_window=ref_window,
                ref_label=ref or "", trusted_windows=trusted_windows,
                sensitivity_pct=max(0.0, min(sens, 0.50)),
                run_sensitivity=bool(run_sens),
                **physical)
            overlay = rep.overlay
        if hasattr(win, "apply_recon_overlay"):
            if hasattr(win, "push_display_undo"):
                win.push_display_undo("AI/tool RLC reconstruction overlay")
            win._recon_overlay = overlay
            win.apply_recon_overlay()
        return rep.text
    return f"Unknown tool: {name}"


def _x_unit(win) -> str:
    label = getattr(win, "ed_xlabel", None)
    text = label.text() if label is not None else ""
    m = re.search(r"\(([^)]+)\)", text)
    return m.group(1) if m else "x-units"


# ------------------------------------------------------------- formatting --
def _set_line(win, attr: str, value) -> bool:
    w = getattr(win, attr, None)
    if w is None:
        return False
    w.setText(str(value))
    if hasattr(w, "editingFinished"):
        w.editingFinished.emit()
    return True


def apply_actions(win, actions: list[dict]) -> tuple[list[str], list[str]]:
    applied: list[str] = []
    tool_msgs: list[str] = []
    chans_changed = False
    undo_pushed = False

    def _push_once():
        nonlocal undo_pushed
        if not undo_pushed and hasattr(win, "push_display_undo"):
            win.push_display_undo("AI plot/display action")
            undo_pushed = True

    for act in actions:
        if not isinstance(act, dict):
            continue
        if "run" in act:
            try:
                tool_msgs.append(run_tool(win, act))
            except Exception as e:
                tool_msgs.append(f"Tool {act.get('run')} failed: {e}")
            continue
        for key, val in act.items():
            if key in {
                "set_title", "set_xlabel", "set_ylabel_left",
                "set_ylabel_right", "set_xscale", "set_xrange",
                "set_yrange_left", "set_yrange_right", "align_zeros",
                "lowpass_filter", "top_axis", "channel",
            }:
                _push_once()
            try:
                if key == "set_title" and _set_line(win, "ed_title", val):
                    _push_once()
                    applied.append(f"title -> \u201c{val}\u201d")
                elif key == "set_xlabel" and _set_line(win, "ed_xlabel", val):
                    _push_once()
                    applied.append(f"x-label -> \u201c{val}\u201d")
                elif key == "set_ylabel_left" and \
                        _set_line(win, "ed_yllabel", val):
                    _push_once()
                    applied.append(f"y-left label -> \u201c{val}\u201d")
                elif key == "set_ylabel_right" and \
                        _set_line(win, "ed_yrlabel", val):
                    _push_once()
                    applied.append(f"y-right label -> \u201c{val}\u201d")
                elif key == "set_xscale" and hasattr(win, "spn_xscale"):
                    _push_once()
                    win.spn_xscale.setValue(float(val))
                    applied.append(f"x-scale x{val}")
                elif key == "set_xrange":
                    _push_once()
                    win.pi.vb.setXRange(float(val["min"]),
                                        float(val["max"]), padding=0)
                    applied.append(f"x-range {val['min']}-{val['max']}")
                elif key == "set_yrange_left":
                    _push_once()
                    win.pi.vb.setYRange(float(val["min"]),
                                        float(val["max"]), padding=0)
                    applied.append(f"y-left {val['min']}-{val['max']}")
                elif key == "set_yrange_right" and hasattr(win, "vb_right"):
                    _push_once()
                    if hasattr(win, "chk_zero"):
                        win.chk_zero.setChecked(False)
                    win.vb_right.setYRange(float(val["min"]),
                                           float(val["max"]), padding=0)
                    applied.append(f"y-right {val['min']}-{val['max']}")
                elif key == "align_zeros" and hasattr(win, "chk_zero"):
                    _push_once()
                    win.chk_zero.setChecked(bool(val))
                    applied.append(f"align zeros -> {bool(val)}")
                elif key == "lowpass_filter":
                    _push_once()
                    if hasattr(win, "chk_filter"):
                        win.chk_filter.setChecked(bool(val.get("enabled", True)))
                    if "cutoff_hz" in val and hasattr(win, "spn_filter_hz"):
                        win.spn_filter_hz.setValue(float(val["cutoff_hz"]))
                    if "target" in val and hasattr(win, "cmb_filter_target"):
                        target_map = {
                            "all": "All enabled channels",
                            "left": "Left-axis channels",
                            "current": "Current-like channels",
                        }
                        label = target_map.get(str(val["target"]).lower())
                        idx = win.cmb_filter_target.findText(label) if label else -1
                        if idx >= 0:
                            win.cmb_filter_target.setCurrentIndex(idx)
                    applied.append("low-pass filter updated")
                elif key == "top_axis":
                    _push_once()
                    if hasattr(win, "chk_top"):
                        win.chk_top.setChecked(bool(val.get("on", True)))
                    if "scale" in val and hasattr(win, "spn_topscale"):
                        win.spn_topscale.setValue(float(val["scale"]))
                    if "label" in val:
                        _set_line(win, "ed_toplabel", val["label"])
                    applied.append("top axis updated")
                elif key == "channel":
                    _push_once()
                    name = str(val.get("name", ""))
                    ch = next((c for c in win.channels if c.name == name),
                              None)
                    if ch is None:
                        applied.append(f"channel \u201c{name}\u201d not found")
                        continue
                    if "enabled" in val:
                        ch.enabled = bool(val["enabled"])
                    if "axis" in val and val["axis"] in ("left", "right"):
                        ch.axis = val["axis"]
                    if "gain" in val:
                        ch.gain = float(val["gain"])
                    if "offset" in val:
                        ch.offset = float(val["offset"])
                    if "label" in val:
                        ch.label = str(val["label"])
                    if "formula" in val:
                        ch.formula = str(val["formula"])
                    chans_changed = True
                    applied.append(f"channel {name} updated")
            except Exception as e:
                applied.append(f"{key} failed: {e}")

    try:
        if chans_changed:
            win._transform_cache.clear()
            win._rebuild_table()
        if applied:
            win.refresh_plot()
    except Exception as e:
        applied.append(f"refresh failed: {e}")
    return applied, tool_msgs


def process_reply(win, reply: str) -> tuple[str, list[str], list[str]]:
    clean, actions = extract_actions(reply)
    if not actions:
        return clean, [], []
    applied, tool_msgs = apply_actions(win, actions)
    return clean, applied, tool_msgs
