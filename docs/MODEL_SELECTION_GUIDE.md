# Choosing a Local AI Model

Scope Studio separates numerical analysis from language-model assistance.
NumPy/SciPy computes values; the LLM explains results and sends whitelisted UI
actions such as changing labels, palettes, filters, and journal presets.

This means the best model is not necessarily the largest model. Choose the
smallest model that can reliably follow instructions on your device.

## Recommended Tiers

### Mac direct MLX — Full default

- Profile in app: `Mac direct MLX - Qwen3.5 4B 4-bit`
- Backend: MLX direct
- Model name: `mlx-community/Qwen3.5-4B-MLX-4bit`
- Best for: the shipped Full desktop assistant on Apple Silicon
- Install:
  `pip install -r requirements-mlx-mac.txt`
- Benchmark basis: #1 in the Scope Studio MLX benchmark
  (`backtests/mlx_model_benchmark_report.txt`), final 8.79/10,
  tool/coder PASS, light-default PASS.
- Local folder use: MLX models are folders containing files such as
  `config.json`, `tokenizer.json`, and `.safetensors`. GGUF files belong to
  the llama.cpp backend. The app auto-scans a plugged-in model vault at
  `/Volumes/<drive>/Models/mlx`, `/Volumes/<drive>/ScopeStudioModels/mlx`,
  `/Volumes/<drive>/models/mlx`, or `/Volumes/<drive>/mlx`, then falls back
  to `~/models/mlx`. Jean's current vault is
  `/Volumes/JeanDrive1/Models/mlx`.
- Tradeoff: macOS/Apple Silicon focused; use Ollama or llama.cpp on other
  platforms.

### Mac Lite MLX

- Profile in app: `Mac Lite MLX - Llama 3.2 3B 4-bit`
- Backend: MLX direct
- Model name: `mlx-community/Llama-3.2-3B-Instruct-4bit`
- Best for: the Lite build or smaller laptops
- Benchmark basis: 1.7 GB, 2.9 s/task, tool/coder PASS, light-default PASS.
- Tradeoff: faster and smaller than the Full default, but weaker for deeper
  scientific interpretation.

### Mac Pro analyst MLX

- Profile in app: `Mac Pro analyst MLX - Qwen3 14B 4-bit`
- Backend: MLX direct
- Model name: `mlx-community/Qwen3-14B-4bit`
- Best for: optional deeper reasoning when the user has enough memory
- Benchmark basis: only model in the Scope Studio MLX benchmark with analyst
  PASS; also best personal-assistant benchmark score.
- Tradeoff: larger and slower, so it is optional rather than the default.
- Storage note: this model has a shard larger than FAT32's single-file limit.
  Store it on APFS or exFAT. On a FAT32 drive such as
  `/Volumes/JEAN D2`, ship only the required Full/Lite models unless the
  model vault is moved to a compatible filesystem.

### Shipping model vault

The shipping set is defined in `config/shipping_models.json`, not by whatever
happens to be in `~/models`. To sync/download only those models:

```bash
./scripts/download_mlx_models.sh
INCLUDE_OPTIONAL=1 ./scripts/download_mlx_models.sh   # add Pro analyst model
```

When a drive is mounted, the scripts and app prefer:

- `/Volumes/<drive>/Models/mlx`
- `/Volumes/<drive>/ScopeStudioModels/mlx`
- `/Volumes/<drive>/Scope Studio Models/mlx`
- `/Volumes/<drive>/models/mlx`
- `/Volumes/<drive>/mlx`

### Tiny router

- Profile in app: `Tiny router - FunctionGemma 270M`
- Backend: Ollama
- Model name: `functiongemma`
- Best for: plot actions, structured commands, fast local UI control
- Tradeoff: not intended for deep scientific discussion
- Source:
  https://ollama.com/library/functiongemma

### Lightweight chat

- Profile in app: `Light chat - Llama 3.2 1B`
- Backend: Ollama
- Model name: `llama3.2:1b`
- Best for: basic explanations, rewriting captions, simple plot edits
- Tradeoff: weaker instruction following than 3B/7B models
- Source:
  https://ollama.com/library/llama3.2

### Balanced chat/action

- Profile in app: `Balanced chat/action - Llama 3.2 3B`
- Backend: Ollama
- Model name: `llama3.2`
- Best for: better tool use, summaries, student-friendly explanations
- Tradeoff: more RAM and slightly more latency than 1B
- Source:
  https://ollama.com/library/llama3.2

### Mac heavy fallback

- Profile in app: `Mac heavy - Qwen 7B GGUF Q4`
- Backend: llama.cpp through `llama-cpp-python`
- Example model path:
  `~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf`
- Best for: richer analysis and longer discussions on M-series MacBook Pro
- Tradeoff: larger download and more RAM
- Sources:
  https://github.com/ggml-org/llama.cpp
  https://github.com/abetlen/llama-cpp-python

## Device-Based Recommendation

- 8 GB RAM laptop: start with `functiongemma` or `llama3.2:1b`.
- 16 GB RAM laptop: start with `llama3.2:1b`; try `llama3.2` if latency is acceptable.
- Apple Silicon MacBook Pro, 18-24 GB RAM: try MLX direct first; use
  Ollama MLX models or a 7B GGUF model as fallbacks for heavier analysis.
- Shared lab workstation: benchmark several models and set a lab default.

## Benchmark Method

Run:

```bash
python scripts/benchmark_models.py
```

The benchmark records:

- model profile
- backend
- response time
- whether a valid JSON action was returned
- whether the model followed a plotting instruction

Use the fastest model that reliably produces correct actions for your common
workflow. A smaller model that changes the plot correctly is better than a
larger model that gives beautiful but unusable prose.

## Safety Rule

Scope Studio does not let the LLM execute arbitrary code or rewrite project
files. The LLM can only request whitelisted actions or create inactive draft
tools in `tool_sandbox/drafts/`. Draft tools run on in-memory arrays only and
must pass deterministic tests before promotion. This is intentional for
open-source use in student labs.
