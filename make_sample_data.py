"""make_sample_data.py — Generates a scope-like CSV that mimics the pulsed
TF-coil current-driver shots in the slides: Pearson monitor, S1/S2 bus-bar
measurements, total current, and a PWM control signal. Includes an
instrument-style preamble to test the loader. ~2M rows by default."""
import numpy as np

N = 2_000_000
t = np.linspace(0, 5e-3, N)                      # 5 ms record, seconds
f_pwm = 1000.0                                   # 1 kHz pulses
pwm = ((t * f_pwm) % 1.0 < 0.25).astype(float)   # 25% duty
ctrl = pwm * 5.0                                 # 0–5 V control signal

rng = np.random.default_rng(7)


def pulse_train(amp, tau_r, tau_f, noise):
    y = np.zeros(N)
    period = 1.0 / f_pwm
    ph = (t % period)
    on = ph < 0.25 * period
    y[on] = amp * (1 - np.exp(-ph[on] / tau_r))
    off = ~on
    y[off] = amp * (1 - np.exp(-0.25 * period / tau_r)) * \
        np.exp(-(ph[off] - 0.25 * period) / tau_f)
    return y + rng.normal(0, noise, N)


i_s1 = pulse_train(220, 8e-5, 2.2e-4, 1.5)       # bus-bar S1 (A)
i_s2 = pulse_train(235, 8e-5, 2.2e-4, 1.5)       # bus-bar S2 (A)
# EMP-like spikes on the bus-bar channel near switching edges
edges = np.where(np.abs(np.diff(pwm)) > 0)[0]
for e in edges[::2]:
    k = slice(e, min(e + 400, N))
    i_s2[k] += 60 * np.exp(-np.arange(k.stop - k.start) / 80.0) * \
        np.sin(2 * np.pi * 138e3 * t[k])
pearson_v = (i_s1 + i_s2) / 1000.0 + rng.normal(0, 0.002, N)  # 0.001 V/A

data = np.column_stack([t, pearson_v, i_s1 / 100.0, i_s2 / 100.0, ctrl])
hdr = ("Model,DSOX1204G\nSerial,CN61234567\nPoints,%d\n"
       "Time (s),Pearson (V),S1 busbar (V),S2 busbar (V),Control (V)\n" % N)
with open("sample_shot.csv", "w") as f:
    f.write(hdr)
    np.savetxt(f, data, delimiter=",", fmt="%.7g")
print("wrote sample_shot.csv:", N, "rows")
