#!/usr/bin/env node
/* Backtest PulseLab's engine against real Tektronix shots and against
 * the known ground truth established by Scope Studio's Python pipeline
 * (Change Set 16): raw CH2 max 6620 A on the 6.6 kA shot; baseline
 * offset ~ +65 A; step-series plateaus. Exits non-zero on failure. */
"use strict";
const fs = require("fs");
const path = require("path");
const P = require("./pulselab_core.js");

const args = process.argv.slice(2);
const DATA = args[0] || path.join(process.env.HOME || "", "Documents", "Data Scope");
const out = [];
function log(s) { out.push(s); console.log(s); }
function approx(a, b, tolPct, what) {
  const err = Math.abs(a - b) / Math.abs(b) * 100;
  const ok = err <= tolPct;
  log(`  ${ok ? "PASS" : "FAIL"} ${what}: ${a.toFixed(1)} vs ${b} ` +
      `(${err.toFixed(2)}% err, tol ${tolPct}%)`);
  if (!ok) process.exitCode = 1;
  return ok;
}

log("PulseLab engine backtest — " + new Date().toISOString());

// Real shot CSVs live on the user's machine, not in CI. When they're
// absent we SKIP the calibrated-benchmark sections (don't crash) so the
// data-independent engine checks ([3]-[5]) still run as a CI sanity gate.
let skippedRealData = 0;
function readShot(file, label) {
  if (!fs.existsSync(file)) {
    log(`\n${label} ${file}\n  SKIP (file not present — real shot data ` +
        `is local-only; engine checks below still run)`);
    skippedRealData++;
    return null;
  }
  return P.parseScopeCSV(fs.readFileSync(file, "utf8"));
}

// ---- shot 1: the 6.6 kA benchmark -----------------------------------
const f1 = path.join(DATA,
  "2026-04-20 4 Modules full amperage @ 100% 6.6kA", "T0000.CSV");
const t0 = Date.now();
const d1 = readShot(f1, "[1]");
if (d1) {
  log("[1] " + f1);
  log(`  parsed ${d1.nRows.toLocaleString()} rows x ${d1.names.length} cols ` +
      `in ${Date.now() - t0} ms; units: ` + JSON.stringify(d1.units));
  if (d1.names.join() !== "TIME,CH1,CH1 Peak Detect,CH2,CH2 Peak Detect") {
    log("  FAIL header mismatch: " + d1.names.join());
    process.exitCode = 1;
  } else log("  PASS header + preamble detection");
  const time = d1.cols[0], ch2 = d1.cols[3];
  const s2 = P.stats(ch2);
  approx(s2.peak, 6620, 0.5, "CH2 raw peak vs 6.6 kA nameplate");
  const off = P.preTriggerOffset(time, ch2);
  approx(off, 65.2, 8, "CH2 pre-trigger baseline offset");
  const dec = P.minmaxDecimate(time, ch2, 4000);
  const sd = P.stats(dec.y);
  approx(sd.peak, s2.peak, 0.01, "decimation preserves the peak exactly");
  log(`  decimated ${d1.nRows.toLocaleString()} -> ${dec.x.length} points`);
}

// ---- shot 2: a step-series file --------------------------------------
const f2 = path.join(DATA,
  "2026-04-09 4 Modules in parallel 100% and step current waaveforms",
  "T0012.CSV");
const d2 = readShot(f2, "[2]");
if (d2) {
  log("[2] " + f2);
  log(`  parsed ${d2.nRows.toLocaleString()} rows x ${d2.names.length} cols`);
  const i2 = d2.names.indexOf("CH2");
  const st2 = P.stats(d2.cols[i2]);
  approx(st2.peak, 646, 2, "T0012 CH2 peak vs Python pipeline (645.96)");
}

// ---- spike detector sanity (synthetic, known truth) -------------------
log("\n[3] synthetic spike-detector check");
const N = 50000, xs = [], ys = [];
let seed = 42;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5;
for (let i = 0; i < N; i++) { xs.push(i * 0.01); ys.push(rnd() * 2); }
ys[12000] += 60; ys[30000] -= 45;            // two big spikes
const ev = P.detectSpikes(xs, ys, 6);
log(`  detected ${ev.length} event(s); strongest at t=${ev[0].t.toFixed(2)}`);
const hit = ev.some(e => Math.abs(e.t - 120) < 1) &&
            ev.some(e => Math.abs(e.t - 300) < 1);
log(`  ${hit ? "PASS" : "FAIL"} both injected spikes found`);
if (!hit) process.exitCode = 1;

// ---- bundled surfaces generate --------------------------------------
log("\n[4] bundled surface datasets");
for (const gen of P.surfaces) {
  const s = gen();
  const flat = s.z.flat().filter(v => isFinite(v));
  log(`  ${s.name}: ${s.z.length} x ${s.z[0].length}, ` +
      `z in [${Math.min(...flat).toFixed(2)}, ${Math.max(...flat).toFixed(2)}]  OK`);
}

// ---- stride decimation + V-I map (overlay / V-I modes) ---------------
log("\n[5] strideDecimate + viMap (3D overlay / V-I map engine)");
{
  const n = 9000, sx = [], sy = [];
  for (let i = 0; i < n; i++) { sx.push(i * 0.001); sy.push(Math.sin(i * 0.01)); }
  const sd = P.strideDecimate(sx, sy, 3000);
  const okStride = sd.x.length === 3000 && sd.x[0] === sx[0] &&
                   sd.x[1] === sx[3];                 // step = floor(9000/3000)=3
  log(`  ${okStride ? "PASS" : "FAIL"} strideDecimate ${n} -> ${sd.x.length} ` +
      `(step 3, endpoints preserved)`);
  if (!okStride) process.exitCode = 1;

  // V-I: a clean circle (x=cos, y=sin) -> locus stays on the unit circle,
  // color rises monotonically 0..1 with sample index (time).
  const cx = [], cy = [];
  const M = 4000;
  for (let i = 0; i < M; i++) {
    const th = 2 * Math.PI * i / M; cx.push(Math.cos(th)); cy.push(Math.sin(th));
  }
  const vi = P.viMap(cx, cy);
  const radii = vi.x.map((x, i) => Math.hypot(x, vi.y[i]));
  const maxRadErr = Math.max(...radii.map(r => Math.abs(r - 1)));
  const colorMono = vi.c[0] === 0 && vi.c[vi.c.length - 1] <= 1 &&
                    vi.c.every((v, i) => i === 0 || v >= vi.c[i - 1]);
  const okVI = vi.n === M && maxRadErr < 1e-9 && colorMono;
  log(`  ${okVI ? "PASS" : "FAIL"} viMap ${vi.n} pts, locus radius err ` +
      `${maxRadErr.toExponential(1)}, color monotonic ${colorMono}`);
  if (!okVI) process.exitCode = 1;

  // NaN-aware: viMap drops non-finite pairs
  const nanVi = P.viMap([1, NaN, 3], [4, 5, NaN]);
  const okNaN = nanVi.n === 1 && nanVi.x[0] === 1 && nanVi.y[0] === 4;
  log(`  ${okNaN ? "PASS" : "FAIL"} viMap drops NaN pairs (kept ${nanVi.n}/3)`);
  if (!okNaN) process.exitCode = 1;
}

log("\nRESULT: " + (process.exitCode ? "FAILURES PRESENT" : "ALL PASS") +
    (skippedRealData ? ` (${skippedRealData} real-data section(s) skipped)` : ""));
fs.writeFileSync(path.join(__dirname, "backtest_report.txt"),
                 out.join("\n") + "\n");
