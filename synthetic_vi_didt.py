"""
synthetic_vi_didt.py - deterministic V/I/di-dt datasets for Scope Studio.

These files are educational test data for the 3D plotter and local assistant.
They are not experimental measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd


L_INTERNAL_H = 166e-6
R_LOAD_OHM = 0.012
T_END_S = 0.120
PULSE_START_S = 0.005
PULSE_END_S = 0.080


@dataclass(frozen=True)
class SyntheticConfig:
    inductance_h: float = L_INTERNAL_H
    resistance_ohm: float = R_LOAD_OHM
    t_end_s: float = T_END_S
    dt_s: float = 50e-6
    pulse_start_s: float = PULSE_START_S
    pulse_end_s: float = PULSE_END_S
    drive_voltages: tuple[float, ...] = (20.0, 40.0, 60.0, 80.0)
    seed: int = 166


def simulate_trace(voltage: float, cfg: SyntheticConfig):
    t = np.arange(0.0, cfg.t_end_s + 0.5 * cfg.dt_s, cfg.dt_s)
    on = (t >= cfg.pulse_start_s) & (t <= cfg.pulse_end_s)
    # Smooth edges avoid a synthetic derivative impulse that dominates
    # teaching surfaces.
    rise = 1.0 / (1.0 + np.exp(-(t - cfg.pulse_start_s) / 0.00045))
    fall = 1.0 / (1.0 + np.exp((t - cfg.pulse_end_s) / 0.00065))
    gate = rise * fall
    ripple = 0.015 * np.sin(2 * np.pi * 2_000 * t) * on
    v_drive = voltage * gate * (1.0 + ripple)

    current = np.zeros_like(t)
    didt = np.zeros_like(t)
    for k in range(1, t.size):
        didt[k - 1] = (
            v_drive[k - 1] - cfg.resistance_ohm * current[k - 1]
        ) / cfg.inductance_h
        current[k] = current[k - 1] + didt[k - 1] * cfg.dt_s
    didt[-1] = (v_drive[-1] - cfg.resistance_ohm * current[-1]) \
        / cfg.inductance_h
    l_didt = cfg.inductance_h * didt

    rng = np.random.default_rng(cfg.seed + int(voltage))
    pearson = current + rng.normal(0.0, max(voltage, 1.0) * 0.015, t.size)
    # Matches the "BBCM" preset: gain=4, formula=(2.52-x)*750.
    bbcm_v = 2.52 - current / (750.0 * 4.0)
    control = 5.0 * gate
    return {
        "time_s": t,
        "time_ms": t * 1000.0,
        "drive_voltage_V": np.full_like(t, voltage),
        "drive_waveform_V": v_drive,
        "current_A": current,
        "pearson_current_A": pearson,
        "bbcm_voltage_V": bbcm_v,
        "control_V": control,
        "dI_dt_A_per_s": didt,
        "L_didt_V": l_didt,
    }


def make_surface_dataframe(cfg: SyntheticConfig, z_name: str, c_name: str):
    rows = []
    for voltage in cfg.drive_voltages:
        tr = simulate_trace(voltage, cfg)
        rows.append(pd.DataFrame({
            "x": tr["time_ms"],
            "y": tr["drive_voltage_V"],
            "z": tr[z_name],
            "c": tr[c_name],
            "time_ms": tr["time_ms"],
            "drive_voltage_V": tr["drive_voltage_V"],
            z_name: tr[z_name],
            c_name: tr[c_name],
        }))
    return pd.concat(rows, ignore_index=True)


def write_examples(folder: str) -> list[str]:
    cfg = SyntheticConfig()
    os.makedirs(folder, exist_ok=True)
    written: list[str] = []

    # Scope-style single-shot file at the 80 V benchmark level.
    tr = simulate_trace(80.0, cfg)
    scope = pd.DataFrame({
        "TIME": tr["time_s"],
        "CH1 BBCM Voltage (V)": tr["bbcm_voltage_V"],
        "CH2 Pearson Current (A)": tr["pearson_current_A"],
        "CH3 Control (V)": tr["control_V"],
        "CH4 dI/dt (A/s)": tr["dI_dt_A_per_s"],
        "CH5 L*dI/dt (V)": tr["L_didt_V"],
        "CH6 Drive Voltage (V)": tr["drive_waveform_V"],
    })
    scope_path = os.path.join(folder, "synthetic_vi_didt_scope.csv")
    with open(scope_path, "w", encoding="utf-8") as handle:
        handle.write("Synthetic,Scope Studio V/I/di-dt demo\n")
        handle.write(f"Internal Inductance H,{cfg.inductance_h:.9g}\n")
        handle.write(f"Load Resistance Ohm,{cfg.resistance_ohm:.9g}\n")
        handle.write("Original Data,Synthetic demonstration only\n")
        scope.to_csv(handle, index=False)
    written.append(scope_path)

    specs = [
        ("synthetic_vi_didt_surface_current.csv",
         "current_A", "dI_dt_A_per_s"),
        ("synthetic_vi_didt_surface_didt.csv",
         "dI_dt_A_per_s", "current_A"),
        ("synthetic_vi_didt_surface_voltage.csv",
         "L_didt_V", "current_A"),
    ]
    for name, z_name, c_name in specs:
        path = os.path.join(folder, name)
        make_surface_dataframe(cfg, z_name, c_name).to_csv(path, index=False)
        written.append(path)

    readme = os.path.join(folder, "synthetic_vi_didt_README.txt")
    with open(readme, "w", encoding="utf-8") as handle:
        handle.write(
            "Synthetic V/I/di-dt examples for Scope Studio\n"
            "================================================\n\n"
            "These files are deterministic educational data, not lab data.\n"
            f"L = {cfg.inductance_h:g} H (166 uH)\n"
            f"R = {cfg.resistance_ohm:g} ohm\n"
            "Model: dI/dt = (V_drive(t) - R*I) / L\n"
            "Derived voltage: L*dI/dt\n\n"
            "Surface files use long format:\n"
            "x = time_ms, y = drive_voltage_V, z = selected value, "
            "c = color-encoded companion value.\n"
        )
    written.append(readme)
    return written


def synthetic_current_surface():
    cfg = SyntheticConfig()
    df = make_surface_dataframe(cfg, "current_A", "dI_dt_A_per_s")
    piv_z = df.pivot_table(index="y", columns="x", values="z",
                           aggfunc="mean")
    piv_c = df.pivot_table(index="y", columns="x", values="c",
                           aggfunc="mean")
    x, y = np.meshgrid(piv_z.columns.to_numpy(float),
                       piv_z.index.to_numpy(float))
    return (x, y, piv_z.to_numpy(float)), piv_c.to_numpy(float)


def synthetic_didt_surface():
    cfg = SyntheticConfig()
    df = make_surface_dataframe(cfg, "dI_dt_A_per_s", "current_A")
    piv_z = df.pivot_table(index="y", columns="x", values="z",
                           aggfunc="mean")
    piv_c = df.pivot_table(index="y", columns="x", values="c",
                           aggfunc="mean")
    x, y = np.meshgrid(piv_z.columns.to_numpy(float),
                       piv_z.index.to_numpy(float))
    return (x, y, piv_z.to_numpy(float)), piv_c.to_numpy(float)


def synthetic_voltage_surface():
    cfg = SyntheticConfig()
    df = make_surface_dataframe(cfg, "L_didt_V", "current_A")
    piv_z = df.pivot_table(index="y", columns="x", values="z",
                           aggfunc="mean")
    piv_c = df.pivot_table(index="y", columns="x", values="c",
                           aggfunc="mean")
    x, y = np.meshgrid(piv_z.columns.to_numpy(float),
                       piv_z.index.to_numpy(float))
    return (x, y, piv_z.to_numpy(float)), piv_c.to_numpy(float)
