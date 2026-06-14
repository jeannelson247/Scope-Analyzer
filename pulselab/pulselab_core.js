/*
 * pulselab_core.js — deterministic analysis engine for PulseLab.
 * Pure functions, no DOM: runs identically in the browser and in Node
 * (which is how the backtest verifies it against real scope files).
 *
 * Design rule inherited from Scope Studio: the AI never computes —
 * everything numeric lives here.
 */
(function (root) {
  "use strict";

  // ---------- CSV parsing (Tektronix/Rigol preambles, any delimiter) ----
  function sniffDelim(text) {
    const head = text.slice(0, 4000);
    const counts = [",", ";", "\t"].map(d => [d, head.split(d).length]);
    counts.sort((a, b) => b[1] - a[1]);
    return counts[0][0];
  }

  function isNum(tok) {
    if (tok === undefined || tok === null) return false;
    const s = String(tok).trim();
    if (!s) return false;
    return !isNaN(Number(s));
  }

  function parseScopeCSV(text) {
    const delim = sniffDelim(text);
    const lines = text.split(/\r?\n/);
    const meta = {};
    let start = -1, header = null;

    for (let i = 0; i < Math.min(lines.length, 80); i++) {
      const toks = lines[i].split(delim);
      const vals = toks.filter(t => t.trim() !== "");
      const allNum = vals.length >= 2 && isNum(toks[0]) && vals.every(isNum);
      if (allNum) {
        // next non-empty line must also be numeric with same field count
        let j = i + 1;
        while (j < lines.length && !lines[j].trim()) j++;
        if (j < lines.length) {
          const t2 = lines[j].split(delim);
          const v2 = t2.filter(t => t.trim() !== "");
          if (!(v2.length >= 2 && isNum(t2[0]) && v2.every(isNum) &&
                t2.length === toks.length)) continue;
        }
        start = i;
        // look back for a header row with same field count, mostly text
        for (let k = i - 1; k >= Math.max(0, i - 4); k--) {
          const ht = lines[k].split(delim).map(s => s.trim());
          const hv = ht.filter(Boolean);
          if (ht.length === toks.length && hv.length &&
              hv.filter(isNum).length / hv.length < 0.5) {
            header = ht.map((h, c) => h || "col" + c);
            break;
          }
        }
        break;
      }
      // preamble "Key,val,..." line
      if (toks[0] && !isNum(toks[0])) meta[toks[0].trim()] = toks.slice(1);
    }
    if (start < 0) throw new Error("No numeric data rows found.");

    const ncol = lines[start].split(delim).length;
    const names = header ||
      ["TIME"].concat(Array.from({length: ncol - 1}, (_, i) => "CH" + (i + 1)));
    const cols = names.map(() => []);
    for (let i = start; i < lines.length; i++) {
      const ln = lines[i];
      if (!ln || !ln.trim()) continue;
      const toks = ln.split(delim);
      if (!isNum(toks[0])) continue;
      for (let c = 0; c < names.length; c++) {
        const v = Number(toks[c]);
        cols[c].push(isNaN(v) ? NaN : v);
      }
    }
    // units from Tektronix 'Vertical Units' row (aligns after time col)
    const units = {};
    const vu = meta["Vertical Units"];
    if (vu) for (let c = 1; c < names.length; c++) {
      if (vu[c - 1]) units[names[c]] = vu[c - 1];
    }
    return { names, cols, meta, units, nRows: cols[0].length };
  }

  // ---------- decimation (min-max envelope: spikes survive) -------------
  function minmaxDecimate(x, y, target) {
    const n = x.length;
    if (n <= target) return { x: x, y: y };
    const nb = Math.max(4, target >> 1);
    const xs = [], ys = [];
    for (let b = 0; b < nb; b++) {
      const s = Math.floor(b * n / nb), e = Math.max(s + 1, Math.floor((b + 1) * n / nb));
      let iMin = s, iMax = s;
      for (let i = s; i < e; i++) {
        if (y[i] < y[iMin]) iMin = i;
        if (y[i] > y[iMax]) iMax = i;
      }
      const a = Math.min(iMin, iMax), c = Math.max(iMin, iMax);
      xs.push(x[a], x[c]); ys.push(y[a], y[c]);
    }
    return { x: xs, y: ys };
  }

  // ---------- statistics -------------------------------------------------
  function stats(y) {
    let mn = Infinity, mx = -Infinity, sum = 0, sq = 0, n = 0;
    for (const v of y) {
      if (isNaN(v)) continue;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      sum += v; sq += v * v; n++;
    }
    const mean = sum / n;
    return { peak: mx, min: mn, mean: mean,
             rms: Math.sqrt(sq / n), n: n };
  }

  function preTriggerOffset(x, y) {
    let s = 0, n = 0;
    for (let i = 0; i < x.length; i++) {
      if (x[i] < 0 && !isNaN(y[i])) { s += y[i]; n++; }
    }
    if (n < 16) { // no pre-trigger: first 5%
      const m = Math.max(16, Math.floor(y.length / 20));
      s = 0; n = 0;
      for (let i = 0; i < m; i++) if (!isNaN(y[i])) { s += y[i]; n++; }
    }
    return n ? s / n : 0;
  }

  // ---------- anomaly scan (MAD spike detection) -------------------------
  function median(arr) {
    const a = Array.from(arr).filter(v => !isNaN(v)).sort((p, q) => p - q);
    const m = a.length >> 1;
    return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
  }

  function detectSpikes(x, y, sigma) {
    sigma = sigma || 6;
    // residual vs moving mean
    const w = Math.max(5, Math.floor(y.length / 500)) | 1;
    const res = new Array(y.length).fill(0);
    let acc = 0;
    const q = [];
    for (let i = 0; i < y.length; i++) {
      q.push(y[i]); acc += y[i];
      if (q.length > w) acc -= q.shift();
      res[i] = y[i] - acc / q.length;
    }
    const mad = median(res.map(Math.abs)) * 1.4826 || 1e-12;
    const events = [];
    let inEvt = false, best = 0, bestI = 0;
    for (let i = 0; i < res.length; i++) {
      const z = Math.abs(res[i]) / mad;
      if (z >= sigma) {
        if (!inEvt) { inEvt = true; best = 0; }
        if (z > best) { best = z; bestI = i; }
      } else if (inEvt && z < sigma / 2) {
        events.push({ t: x[bestI], amp: res[bestI], z: best });
        inEvt = false;
      }
    }
    if (inEvt) events.push({ t: x[bestI], amp: res[bestI], z: best });
    events.sort((a, b) => b.z - a.z);
    return events;
  }

  // ---------- stride decimation (preserves parametric/3D shape) ---------
  // Plain every-Nth-sample stride, mirroring surface3d.py's y[::step].
  // Min-max decimation distorts a parametric (V-I) path or a 3D ribbon,
  // so the overlay/V-I modes use this instead of minmaxDecimate.
  function strideDecimate(x, y, target) {
    const n = Math.min(x.length, y.length);
    const step = Math.max(1, Math.floor(n / (target || 3000)));
    if (step === 1) return { x: x.slice(0, n), y: y.slice(0, n) };
    const xs = [], ys = [];
    for (let i = 0; i < n; i += step) { xs.push(x[i]); ys.push(y[i]); }
    return { x: xs, y: ys };
  }

  // ---------- V-I trajectory map (the switching locus) ------------------
  // Ports surface3d.py's _vi_map: Y-channel vs X-channel as a parametric
  // path, colored by time (c in [0,1], dark = early -> bright = late).
  // Spikes that hide in the time domain show as excursions off the locus.
  // NaN-aware; stride-decimated to maxPts (default 200k for the browser).
  function viMap(x, y, maxPts) {
    maxPts = maxPts || 200000;
    const n = Math.min(x.length, y.length);
    const step = Math.max(1, Math.floor(n / maxPts));
    const xs = [], ys = [], c = [];
    const denom = n > 1 ? n - 1 : 1;
    for (let i = 0; i < n; i += step) {
      if (isNaN(x[i]) || isNaN(y[i])) continue;
      xs.push(x[i]); ys.push(y[i]); c.push(i / denom);
    }
    return { x: xs, y: ys, c: c, n: xs.length };
  }

  // ---------- bundled surface datasets (generated, three of them) -------
  function linspace(a, b, n) {
    return Array.from({length: n}, (_, i) => a + (b - a) * i / (n - 1));
  }

  // 1. 3D Gaussian
  function gaussianSurface(n) {
    n = n || 81;
    const ax = linspace(-3, 3, n);
    const z = ax.map(yy => ax.map(xx => Math.exp(-(xx * xx + yy * yy) / 2)));
    return { x: ax, y: ax, z: z, name: "3D Gaussian",
             desc: "exp(-(x²+y²)/2) — the reference bump." };
  }

  // 2. Mexican-hat potential (symmetry breaking)
  function mexicanHatSurface(n) {
    n = n || 101;
    const lim = 1.6;
    const ax = linspace(-lim, lim, n);
    const z = ax.map(yy => ax.map(xx => {
      const r2 = xx * xx + yy * yy;
      return -r2 + r2 * r2;
    }));
    return { x: ax, y: ax, z: z, name: "Mexican-hat potential",
             desc: "V(r) = -r² + r⁴ — spontaneous symmetry breaking." };
  }

  // 3. RLC discharge family: current vs (time, charging voltage) — the
  //    physics of the user's coil driver as a surface
  function rlcSurface(nT, nV) {
    nT = nT || 140; nV = nV || 60;
    const tauR = 2, tauD = 160, R = 0.068;          // ms, ms, ohm
    const tAx = linspace(0, 60, nT);                 // ms
    const vAx = linspace(100, 750, nV);              // charging volts
    const k = 1 / (Math.pow(tauR / tauD, tauR / (tauD - tauR)) -
                   Math.pow(tauR / tauD, tauD / (tauD - tauR)));
    const z = vAx.map(V => tAx.map(t =>
      (V / R) / k > 0
        ? (V / R) * (Math.exp(-t / tauD) - Math.exp(-t / tauR)) /
          (Math.exp(-((tauR * tauD / (tauD - tauR)) * Math.log(tauD / tauR)) / tauD) -
           Math.exp(-((tauR * tauD / (tauD - tauR)) * Math.log(tauD / tauR)) / tauR))
        : 0));
    return { x: tAx, y: vAx, z: z, name: "RLC discharge family",
             xlabel: "Time (ms)", ylabel: "Charging voltage (V)",
             zlabel: "Current (A)",
             desc: "I(t; V₀) for the coil driver (τ_rise 2 ms, τ_droop " +
                   "160 ms, R 68 mΩ) — peak current scales with V₀/R." };
  }

  const api = { parseScopeCSV, minmaxDecimate, strideDecimate, viMap,
                stats, preTriggerOffset, detectSpikes, median,
                surfaces: [gaussianSurface, mexicanHatSurface, rlcSurface] };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PulseLab = api;
})(typeof self !== "undefined" ? self : this);
