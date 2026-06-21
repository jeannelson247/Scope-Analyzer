# Scope Studio — User Guide for Scientists (no coding required)

This guide is written for students and researchers who work with
oscilloscope data but are new to Python. It covers installation, daily
use, calibrating your own current sensors, choosing a local AI model for
your laptop, and tailoring the program to your hardware.

---

## 1. What Scope Studio does

Scope Studio loads oscilloscope CSV exports (Tektronix, Rigol, LeCroy and
similar), plots up to four channels in publication-ready (Nature-style)
figures, computes statistics (peak, minimum, average, RMS), scans for
anomalies (spikes, clipping, drift, module imbalance), and includes an
optional local AI side-chat that runs entirely on your computer — no data
ever leaves your machine.

One design rule explains the whole program: **the AI never computes
numbers.** All statistics come from deterministic NumPy code; the AI only
*reads* those results, *explains* them, and *routes* your requests to the
right tool. We verified this rule empirically — in our benchmark
(`backtests/model_data_benchmark.txt`), every small model scored 0/3 when
asked to compute averages from raw numbers, but up to 6/6 when asked to
read computed stats and route tool calls. Trust the stats panel, not
mental arithmetic from a chatbot.

---

## 2. Installation (macOS, one time only)

You need Python 3.10 or newer. Check by opening Terminal
(Cmd+Space, type "Terminal") and running:

```bash
python3 --version
```

If missing, install from https://www.python.org/downloads/ or with
`brew install python`.

Then, inside the Scope Studio folder:

```bash
cd ~/Desktop/scope_studio03        # adjust to where you put the folder
python3 -m venv venv               # creates a private Python environment
source venv/bin/activate           # activates it — prompt shows (venv)
pip install -r requirements.txt    # installs the libraries (~2 min)
```

A **venv** (virtual environment) is a private toolbox of Python libraries
that lives inside the project folder. You create it ONCE. It persists
between sessions — you never remake it, you only re-activate it.

### Every later session — just two lines

```bash
cd ~/Desktop/scope_studio03 && source venv/bin/activate
python3 app.py
```

Common errors:

| Message | Meaning | Fix |
|---|---|---|
| `pip: command not found` | venv not active | `source venv/bin/activate` |
| `No module named 'numpy'` | wrong Python (system, not venv) | activate venv first |
| `command not found: python3` | Python not installed | install Python 3.10+ |

---

## 3. Daily use

1. **Open CSV/TXT…** — pick your scope export. The loader auto-detects the
   instrument preamble, delimiter, and units; "Peak Detect" envelope
   columns are recognized and left unticked by default.
2. **Tick channels** to plot; assign each to the Left or Right axis.
3. **Stats panel** — peak / min / mean / RMS over whatever you have
   zoomed to. Zoom changes the window; stats follow.
4. **Detect anomalies** — one click scans for spikes (with ringing
   frequency, useful for separating EMI pickup from real current),
   clipping, baseline drift, and S1…S4 module imbalance.
5. **Export** — SVG (editable vector) or JPG at 600 dpi in Nature column
   widths (89 mm / 183 mm).

---

## 4. Calibrating for YOUR current sensors

Scopes export either **already-converted amperes** or **raw volts** that
you must convert. Check the channel unit shown after loading:

* Channel shows **(A)** — the scope did the conversion (e.g. a Pearson
  with the probe factor set on the scope). You need **no formula**; at
  most subtract the standing offset with a baseline preset. Our 6.6 kA
  reference shot is this case: CH2 was exported in amps and matched the
  nameplate within 0.3% with no formula at all.
* Channel shows **(V)** — apply a conversion preset, or write your own.

Formulas use `x` (the raw signal), `t` (time in s), `t_ms`, and helpers
like `baseline()`, `lowpass()`, `movmean()`, `integrate()`. Examples:

```
x * 100.0                                  Pearson 0.01 V/A
(x - 2.5) * 1500 / 2                       busbar 750 A/V centered at 2.5 V
baseline((x - 2.5) * 1500 / 2, t_ms, -1)   same, minus pre-trigger offset
```

To add your sensor permanently, copy any entry in `presets.json`, rename
it, change the formula, save, restart. (`presets.json` is plain text —
open it with any editor.)

**Sanity check your calibration** against a known shot:

```bash
python3 scripts/backtest_real_data.py "path/to/your/folder" \
        --expect-amps 6600 --tolerance 0.05
```

It prints PASS/FAIL per file plus peak, plateau, pulse duration, and the
charge integral. Tip: across a series of capacitor-bank shots the charge
should be roughly constant — if it drifts with the current setpoint, your
conversion factor is suspect.

---

## 5. Choosing a local AI model for your laptop

Install [Ollama](https://ollama.com) (`brew install --cask ollama`), then
pull a model sized to your RAM:

| Your machine | Model | Pull command | Role |
|---|---|---|---|
| Any (even 8 GB) | qwen2.5:0.5b | `ollama pull qwen2.5:0.5b` | fastest router — passed all stats-reading + tool-routing tests |
| 8–16 GB | gemma3:1b or llama3.2:1b | `ollama pull gemma3:1b` | light chat + routing |
| 16 GB | llama3.2 (3B) | `ollama pull llama3.2` | better explanations |
| 24 GB (e.g. M4 Pro) | qwen3.5:9b | `ollama pull qwen3.5:9b` | recommended "interpretation" tier — reasoning over anomalies + papers |
| 24 GB, ambitious | qwen3:14b @ Q4 | `ollama pull qwen3:14b` | deeper physics discussion; close other apps |

Rules of thumb learned the hard way:

* Model weights must fit in roughly **half** your RAM to leave room for
  macOS, the app, and a million-row dataframe. A 23 GB model on a 24 GB
  machine **will freeze the computer**.
* Bigger models do NOT give better numbers — numbers come from NumPy.
  Bigger buys better *explanations*, not better *math*.
* Benchmark on your own data before trusting a model:

```bash
python3 scripts/benchmark_data_analysis.py
```

It auto-detects your installed models and prints every model answer next
to the NumPy ground truth so you can check each one by eye.

---

## 6. "Can I fine-tune the model on my data/papers?"

Almost certainly you want **RAG, not fine-tuning**. Scope Studio's paper
index retrieves passages from the PDFs in your papers folder and hands
them to the model per question — instantly updatable (drop in a new PDF),
no training, no risk of the model memorizing wrong numbers. Fine-tuning
on a laptop is slow, fragile, and for this use case usually makes factual
accuracy *worse*. If you ever need a custom behavior (e.g. your lab's
exact report format), a LoRA fine-tune of a 1–4B model with Apple's
`mlx_lm` is the realistic path on a Mac — but try a better system prompt
and RAG first; they solve 95% of cases.

---

## 7. Tailoring to your hardware (cheat sheet)

| You want to… | Edit | How |
|---|---|---|
| Add a current-sensor conversion | `presets.json` | copy an entry, edit formula |
| Add/replace an AI model | `model_catalog.py` | copy a ModelProfile block |
| Change journal figure style | Style dropdown in app | or `style_presets.py` |
| Change anomaly thresholds | side chat or `detect_anomalies.py` defaults | e.g. `threshold_sigma=5` |
| Verify after changes | `scripts/backtest_real_data.py` | PASS/FAIL vs known shot |

Every change you make: please record it in
`DEVELOPMENT_LOG_SCOPE_STUDIO.txt` (what, why, how verified). This is the
project's lab notebook, and it is what keeps a shared open-source tool
trustworthy.

---

## 8. Getting help

* Errors: copy the exact Terminal message into an issue (or to your AI
  assistant). Screenshots of the full Terminal window work too.
* Wrong-looking numbers: run the backtest script on a shot with a known
  current before suspecting the data.
* See `CONTRIBUTING.md` to propose changes, and
  `docs/MODEL_SELECTION_GUIDE.md` for deeper model notes.
