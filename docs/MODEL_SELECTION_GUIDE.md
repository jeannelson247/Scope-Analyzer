# Choosing a Local AI Model

Scope Studio separates numerical analysis from language-model assistance.
NumPy/SciPy computes values; the LLM explains results and sends whitelisted UI
actions such as changing labels, palettes, filters, and journal presets.

This means the best model is not necessarily the largest model. Choose the
smallest model that can reliably follow instructions on your device.

## Recommended Tiers

### Mac direct MLX

- Profile in app: `Mac direct MLX - Llama 3.2 3B 4-bit`
- Backend: MLX direct
- Model name: `mlx-community/Llama-3.2-3B-Instruct-4bit`
- Best for: fast Apple Silicon response without a separate server
- Install:
  `pip install -r requirements-mlx-mac.txt`
- Local folder use: MLX models are folders containing files such as
  `config.json`, `tokenizer.json`, and `.safetensors`. GGUF files belong to
  the llama.cpp backend. If you want models under `~/models`, download an MLX
  model into `~/models/mlx/<model-name>` and select that folder in the app.
- Tradeoff: macOS/Apple Silicon focused; use Ollama or llama.cpp on other
  platforms.

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
