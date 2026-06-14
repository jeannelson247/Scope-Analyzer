"""
ai_assistant.py — Local-LLM analysis with two interchangeable backends.

Backend "mlx"  (recommended on Apple Silicon)
    Runs MLX-LM directly in-process on Apple Silicon. This is the fastest
    Mac-first path when `mlx-lm` is installed.
    Install:  pip install -r requirements-mlx-mac.txt

Backend "llama.cpp"
    Runs a GGUF model in-process via llama-cpp-python with the Metal GPU
    backend — on an M4 Pro all layers are offloaded to the GPU, so a 7-8B
    Q4_K_M model generates fast while using ~5 GB of your 24 GB.
    Install:  pip install llama-cpp-python
        (if pip compiles from source it auto-enables Metal on Apple Silicon)
    Get a model (one-time), e.g. Qwen2.5-7B-Instruct Q4_K_M:
        huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
            Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir ~/models

Backend "ollama"
    Talks to a local Ollama server (which itself wraps llama.cpp, but as a
    separate process with its own model management).

Only computed statistics are sent — never the raw waveform.
Future: point `analyze_shot` at a Hermes Agent endpoint; the UI is agnostic.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error

LOCAL_MLX_MODELS_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get("SCOPE_STUDIO_MLX_MODELS", "~/models/mlx")))
LOCAL_MODELS_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get("SCOPE_STUDIO_MODELS", "~/models")))
EXTERNAL_MLX_MODELS_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get(
        "SCOPE_STUDIO_EXTERNAL_MLX_MODELS",
        "/Volumes/ScopeStudioModels/mlx",
    )
))
LOCAL_GGUF_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get("SCOPE_STUDIO_GGUF_MODELS", "~/models")))
EXTERNAL_GGUF_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get("SCOPE_STUDIO_EXTERNAL_GGUF_MODELS",
                   "/Volumes/ScopeStudioModels/gguf")))


def _volume_mlx_roots() -> list[str]:
    """Auto-detect MLX model vaults on mounted external drives.

    Preferred layouts on a plugged-in drive:
      /Volumes/<drive>/ScopeStudioModels/mlx
      /Volumes/<drive>/Scope Studio Models/mlx
      /Volumes/<drive>/models/mlx
      /Volumes/<drive>/mlx

    The scan is intentionally shallow so opening the model picker never walks
    an arbitrary external disk.
    """
    roots: list[str] = []
    volumes = "/Volumes"
    try:
        volume_names = sorted(os.listdir(volumes))
    except OSError:
        return roots
    skip = {"Macintosh HD", "Codex Installer"}
    for name in volume_names:
        if name in skip or name.startswith("."):
            continue
        base = os.path.join(volumes, name)
        for rel in (
            "ScopeStudioModels/mlx",
            "Scope Studio Models/mlx",
            "Scope Studio Models",
            "models/mlx",
            "mlx",
        ):
            path = os.path.abspath(os.path.join(base, rel))
            if os.path.isdir(path):
                roots.append(path)
    return roots


def _default_gguf_model() -> str:
    """Prefer the external model vault, with local fallback if unplugged."""
    name = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    for root in (EXTERNAL_GGUF_DIR, LOCAL_GGUF_DIR):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path
    return os.path.join(LOCAL_GGUF_DIR, name)


DEFAULT_GGUF = _default_gguf_model()


def mlx_model_roots() -> list[str]:
    """Canonical MLX search locations.

    Local models stay fast and always available. External-drive models are
    optional and appear automatically when the drive is mounted.
    """
    roots = []
    if os.path.isdir(EXTERNAL_MLX_MODELS_DIR):
        roots.append(EXTERNAL_MLX_MODELS_DIR)
    roots.extend(_volume_mlx_roots())
    roots.append(LOCAL_MLX_MODELS_DIR)
    if os.path.isdir(LOCAL_MODELS_DIR):
        roots.append(LOCAL_MODELS_DIR)
    extra = os.environ.get("SCOPE_STUDIO_EXTRA_MLX_MODELS", "")
    for item in extra.split(os.pathsep):
        item = item.strip()
        if item:
            roots.append(os.path.abspath(os.path.expanduser(item)))
    out: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if root not in seen:
            out.append(root)
            seen.add(root)
    return out


def _is_complete_mlx_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return (
        "config.json" in names
        and any(n.startswith("tokenizer") or n in {"vocab.json", "merges.txt"}
                for n in names)
        and any(n.endswith((".safetensors", ".npz")) for n in names)
        and not any(n.endswith(".incomplete") for n in names)
    )


def _default_mlx_model() -> str:
    """Prefer a small, complete local model for the open-source default.

    Under SCOPE_STUDIO_LITE the candidate list is capped at <=4B so the
    Lite build / small laptops never auto-select a 7-9B model.
    """
    from model_catalog import is_lite
    if is_lite():
        candidates = ("Llama-3.2-3B-Instruct-4bit", "Qwen3.5-4B-MLX-4bit")
    else:
        candidates = (
            "Qwen3.5-4B-MLX-4bit",
            "Llama-3.2-3B-Instruct-4bit",
            "Qwen3-14B-4bit",
            "Qwen2.5-Coder-7B-Instruct-4bit",
            "DeepSeek-R1-Distill-Qwen-7B-4bit",
            "Qwen3.5-9B-MLX-4bit",
        )
    for name in candidates:
        for root in mlx_model_roots():
            path = os.path.join(root, name)
            if _is_complete_mlx_dir(path):
                return path
    return f"mlx-community/{candidates[0]}"


DEFAULT_MLX_MODEL = _default_mlx_model()
# Lite build / small laptops default to a 1B chat model; the full M4-Pro
# default stays the 9B benchmark winner. The user can still override either.
from model_catalog import is_lite as _is_lite
DEFAULT_OLLAMA_MODEL = "llama3.2:1b" if _is_lite() else "qwen3.5:9b-mlx"
DEFAULT_ROUTER_MODEL = "qwen2.5:0.5b"     # fastest perfect A+B scorer
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = (
    "You are an assistant for a power-electronics test engineer working on "
    "tokamak coil current-driver modules. You receive summary statistics of "
    "oscilloscope channels from a shot (peak/min/mean/RMS currents, control "
    "signals). Reply concisely: 1) one-line summary, 2) notable observations "
    "(imbalance between modules, unexpected polarity, spikes implied by "
    "peak >> RMS), 3) anything worth re-checking. Do not invent numbers. "
    "Original CSV files are immutable; only reversible in-memory display "
    "transforms, overlays, and explicitly exported derived files are allowed."
)

_llama_cache: dict = {}
_mlx_cache: dict = {}
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def _token_exists(tokenizer, token: str) -> bool:
    try:
        tok_id = tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        return tok_id is not None and tok_id != unk_id
    except Exception:
        return False


def _mlx_prompt(tokenizer, system_prompt: str, prompt: str) -> str:
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        for kwargs in (
            {"tokenize": False, "add_generation_prompt": True,
             "enable_thinking": False},
            {"tokenize": False, "add_generation_prompt": True},
            {"add_generation_prompt": True, "enable_thinking": False},
            {"add_generation_prompt": True},
        ):
            try:
                text = tokenizer.apply_chat_template(messages, **kwargs)
                if isinstance(text, str) and text.strip():
                    return text
            except (TypeError, ValueError, RuntimeError, Exception):
                continue
    if (_token_exists(tokenizer, "<|im_start|>")
            and _token_exists(tokenizer, "<|im_end|>")):
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return f"System:\n{system_prompt}\n\nUser:\n{prompt}\n\nAssistant:\n"


def _clean_reply(text: str) -> str:
    text = _THINK_RE.sub("", text or "").strip()
    if "Final Answer:" in text:
        text = text.split("Final Answer:", 1)[1].strip()
    if "Final answer:" in text:
        text = text.split("Final answer:", 1)[1].strip()
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def _mlx_chat(prompt: str, model_name: str,
              system_prompt: str = SYSTEM_PROMPT,
              max_tokens: int = 512) -> str:
    try:
        from mlx_lm import load, generate
    except ImportError:
        return ("mlx-lm is not installed. On Apple Silicon run:\n"
                "    pip install -r requirements-mlx-mac.txt\n"
                "This enables the direct MLX backend; Ollama and llama.cpp "
                "remain available as fallbacks.")
    try:
        model_name = resolve_mlx_model(model_name or DEFAULT_MLX_MODEL)
    except ValueError as exc:
        return f"MLX model selection error: {exc}"
    cached = _mlx_cache.get(model_name)
    if cached is None:
        model, tokenizer = load(model_name)
        _mlx_cache.clear()       # keep at most one local model resident
        _mlx_cache[model_name] = (model, tokenizer)
    else:
        model, tokenizer = cached
    chat_prompt = _mlx_prompt(tokenizer, system_prompt, prompt)
    try:
        return _clean_reply(generate(
            model, tokenizer, prompt=chat_prompt,
            max_tokens=max_tokens, verbose=False))
    except TypeError:
        # Older mlx-lm versions may expose a narrower signature.
        return _clean_reply(generate(model, tokenizer, prompt=chat_prompt,
                                     verbose=False))


def _llamacpp_chat(prompt: str, gguf_path: str,
                   system_prompt: str = SYSTEM_PROMPT,
                   max_tokens: int = 512) -> str:
    try:
        from llama_cpp import Llama
    except ImportError:
        return ("llama-cpp-python is not installed. Run:\n"
                "    pip install llama-cpp-python\n"
                "(on Apple Silicon this builds with Metal GPU support "
                "automatically).")
    if not os.path.isfile(gguf_path):
        return (f"GGUF model not found at:\n  {gguf_path}\n\n"
                "Download one, e.g.:\n"
                "  pip install -U huggingface_hub\n"
                "  huggingface-cli download "
                "bartowski/Qwen2.5-7B-Instruct-GGUF "
                "Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir ~/models")
    llm = _llama_cache.get(gguf_path)
    if llm is None:
        llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=-1,      # offload everything to Metal
            n_ctx=4096,
            n_threads=os.cpu_count() or 8,
            flash_attn=True,
            verbose=False,
        )
        _llama_cache.clear()      # keep at most one model in RAM
        _llama_cache[gguf_path] = llm
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.3)
    return out["choices"][0]["message"]["content"].strip()


def _ollama_chat(prompt: str, model: str,
                 system_prompt: str = SYSTEM_PROMPT,
                 timeout: float = 120.0,
                 think: bool | None = None,
                 format_schema: dict | str | None = None) -> str:
    payload = {"model": model, "stream": False,
               # keep the model resident between questions so the side chat
               # never pays the ~10 s reload; idle weights cost no CPU/GPU
               "keep_alive": "30m",
               "options": {"temperature": 0.3},
               "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}]}
    if think is not None:
        # False = skip reasoning tokens (fast routing); True = allow them
        # (interpretation). Omitted entirely for models without the flag.
        payload["think"] = think
    if format_schema is not None:
        # Ollama structured outputs: constrains generation to this JSON
        # schema (or "json") - the model CANNOT emit malformed actions.
        payload["format"] = format_schema
    def _post(body: dict) -> str:
        req = urllib.request.Request(
            OLLAMA_URL, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        return data.get("message", {}).get("content", "").strip()

    try:
        try:
            return _post(payload)
        except urllib.error.HTTPError:
            # older Ollama / model rejecting "think" or "format" -
            # retry once without the optional fields
            payload.pop("think", None)
            payload.pop("format", None)
            return _post(payload)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return ("Ollama not reachable - install from https://ollama.com, "
                f"run `ollama pull {model}`, keep it running, retry.\n"
                f"(Details: {e})")


def ask_model(prompt: str, model: str = "", backend: str = "llama.cpp",
              system_prompt: str = SYSTEM_PROMPT,
              max_tokens: int = 512,
              think: bool | None = None,
              format_schema: dict | str | None = None) -> str:
    """Generic local-model call used by both the one-shot analyzer and the
    side chat. `model` is a GGUF path for llama.cpp, or a model name for
    Ollama. `think`/`format_schema` are Ollama-only refinements."""
    try:
        if backend == "mlx":
            return _mlx_chat(prompt, model or DEFAULT_MLX_MODEL,
                             system_prompt=system_prompt,
                             max_tokens=max_tokens)
        if backend == "llama.cpp":
            return _llamacpp_chat(prompt, model or DEFAULT_GGUF,
                                  system_prompt=system_prompt,
                                  max_tokens=max_tokens)
        return _ollama_chat(prompt, model or DEFAULT_OLLAMA_MODEL,
                            system_prompt=system_prompt,
                            think=think, format_schema=format_schema)
    except Exception as e:                          # never crash the GUI
        return f"AI backend error: {e}"


# JSON schema for plot/tool actions - used with format_schema so routing
# calls are structurally valid by construction (Ollama structured outputs).
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        # "none" lets the router decline: interpretation questions fall
        # through to the heavy model instead of forcing a bogus tool call
        # (schema-constrained decoding would otherwise ALWAYS emit one).
        "run": {"type": "string",
                "enum": ["detect_anomalies", "channel_stats",
                         "estimate_saturation", "reconstruct_rlc",
                         "zero_baseline", "none"]},
        "threshold_sigma": {"type": "number"},
        "stat": {"type": "string"},
        "channel": {"type": "string"},
        "sat_level": {"type": "number"},
        "t_start": {"type": "number"},
        "t_end": {"type": "number"},
        "ref_end": {"type": "number"},
    },
    "required": ["run"],
}

ROUTER_FEWSHOT = (
    "Examples:\n"
    'User: scan for spikes above 5 sigma\n'
    '{"run": "detect_anomalies", "threshold_sigma": 5}\n'
    'User: average the visible current channel\n'
    '{"run": "channel_stats", "stat": "mean"}\n'
    'User: the monitor looks saturated, estimate the real peak current\n'
    '{"run": "estimate_saturation"}\n'
    'User: the busbar saturates at 6000 A, estimate the true peak\n'
    '{"run": "estimate_saturation", "sat_level": 6000}\n'
    'User: reconstruct the full waveform through the saturated part, '
    'sensor limit 6000\n'
    '{"run": "reconstruct_rlc", "sat_level": 6000}\n'
    'User: reconstruct 0 to 145 ms, busbar limit 6000, pearson valid '
    'to 5 ms\n'
    '{"run": "reconstruct_rlc", "sat_level": 6000, "t_start": -5, '
    '"t_end": 145, "ref_end": 5}\n'
    'User: make both signals start at 0 A, remove the offset\n'
    '{"run": "zero_baseline"}\n'
    'User: why would module 3 sag relative to the others?\n'
    '{"run": "none"}\n'
    'User: add a legend and relabel the axes, then rescan for spikes\n'
    '{"run": "none"}\n'
)

def _looks_like_mlx_model_dir(path: str) -> bool:
    """Return True if path looks like a local mlx-lm model folder.

    Typical MLX folders contain:
      - config.json
      - tokenizer.json / tokenizer.model / tokenizer_config.json
      - *.safetensors or *.npz weights
    """
    try:
        if not os.path.isdir(path):
            return False
        names = set(os.listdir(path))
    except OSError:
        return False

    has_config = "config.json" in names
    has_tokenizer = any(
        name in names
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        )
    ) or any(name.startswith("tokenizer") for name in names)

    has_weights = any(
        name.endswith((".safetensors", ".npz"))
        for name in names
    )

    return has_config and has_tokenizer and has_weights


def list_mlx_models(roots: list[str] | None = None) -> list[str]:
    """Find local MLX model folders for the app dropdown.

    Default convention:
      ~/models/mlx/<model-folder>
      /Volumes/ScopeStudioModels/mlx/<model-folder>

    The returned strings are full local folder paths, because mlx_lm.load(...)
    can load directly from those folders.
    """
    roots = roots or mlx_model_roots()

    found: list[str] = []
    seen: set[str] = set()

    def add_if_model(path: str):
        path = os.path.abspath(os.path.expanduser(path))
        if path not in seen and _looks_like_mlx_model_dir(path):
            found.append(path)
            seen.add(path)

    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            continue

        # Root itself may be a model.
        add_if_model(root)

        # Search one and two levels deep. This avoids walking huge folders.
        try:
            level1 = [
                os.path.join(root, name)
                for name in sorted(os.listdir(root))
            ]
        except OSError:
            continue

        for p1 in level1:
            if not os.path.isdir(p1):
                continue
            add_if_model(p1)

            try:
                level2 = [
                    os.path.join(p1, name)
                    for name in sorted(os.listdir(p1))
                ]
            except OSError:
                continue

            for p2 in level2:
                if os.path.isdir(p2):
                    add_if_model(p2)

    return found


def is_hf_model_id(value: str) -> bool:
    """Return True for Hugging Face-style model identifiers."""
    value = (value or "").strip()
    if not value or os.path.isabs(os.path.expanduser(value)):
        return False
    if value.startswith((".", "~")) or os.sep in value and value.startswith(os.sep):
        return False
    return "/" in value and not value.endswith("/")


def resolve_mlx_model(model_name: str | None = None) -> str:
    """Resolve a user MLX model selection to a loadable local folder or HF id.

    Users often choose a parent folder such as ``~/models`` or
    ``/Volumes/<drive>/models``. Direct MLX needs the actual model directory
    containing config/tokenizer/weights, so parent folders are resolved to the
    first complete model inside them. Incomplete folders raise a friendly
    message instead of letting ``mlx_lm.load`` fail deep in the stack.
    """
    raw = (model_name or DEFAULT_MLX_MODEL or "").strip()
    if not raw:
        raw = DEFAULT_MLX_MODEL
    value = os.path.abspath(os.path.expanduser(raw)) if raw.startswith(("~", "/", ".")) else raw

    if is_hf_model_id(value):
        return value

    if os.path.isfile(value):
        raise ValueError(
            "MLX direct expects a model folder, not a file. Use the "
            "llama.cpp backend for .gguf files."
        )

    if os.path.isdir(value):
        if _looks_like_mlx_model_dir(value):
            return os.path.abspath(value)
        models = list_mlx_models([value])
        if models:
            return models[0]
        raise ValueError(
            f"{value} is not a complete MLX model folder. Choose a folder "
            "containing config.json, tokenizer files, and .safetensors/.npz "
            "weights, or plug in the ScopeStudioModels drive."
        )

    if raw == DEFAULT_MLX_MODEL and is_hf_model_id(raw):
        return raw
    raise ValueError(
        f"MLX model not found: {raw}. Plug in the model drive, choose a "
        "complete MLX folder, or use a Hugging Face id such as "
        "mlx-community/Qwen3.5-4B-MLX-4bit."
    )


def list_ollama_models(timeout: float = 2.0) -> list[str]:
    """Names of locally installed Ollama models (for the app's model
    picker). Returns [] when the server is unreachable."""
    try:
        with urllib.request.urlopen(
                "http://localhost:11434/api/tags", timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return sorted(m.get("name", "") for m in data.get("models", [])
                      if m.get("name"))
    except Exception:
        return []


def route_action(user_text: str, model: str = "",
                 backend: str = "ollama") -> str:
    """Fast deterministic routing call: thinking disabled, output
    constrained to ACTION_SCHEMA (Ollama) or few-shot-guided JSON (MLX).
    Returns the model's JSON string."""
    return ask_model(
        ROUTER_FEWSHOT + "User: " + user_text,
        model=model, backend=backend,
        system_prompt="You are a tool router. Reply with one JSON object.",
        max_tokens=128, think=False, format_schema=ACTION_SCHEMA)


# preferred small instruction-followers for the MLX router seat, in
# order (CS45 benchmark winners); first one present on disk is used
_MLX_ROUTER_PREFS = ("Qwen3.5-4B-MLX-4bit", "Llama-3.2-3B-Instruct-4bit",
                     "Qwen3.5-4B-4bit", "Qwen2.5-Coder-3B-Instruct-4bit")


def default_mlx_router() -> str:
    """Path of the best installed small MLX model for routing, or ''
    when none is available (caller then skips routing)."""
    try:
        installed = list_mlx_models()
    except Exception:
        return ""
    by_name = {os.path.basename(p.rstrip("/")): p for p in installed}
    for pref in _MLX_ROUTER_PREFS:
        if pref in by_name:
            return by_name[pref]
    return ""


def analyze_shot(stats_text: str, model: str = "",
                 backend: str = "llama.cpp") -> str:
    prompt = "Shot statistics over the current view window:\n\n" + stats_text
    return ask_model(prompt, model=model, backend=backend,
                     system_prompt=SYSTEM_PROMPT, max_tokens=512)
