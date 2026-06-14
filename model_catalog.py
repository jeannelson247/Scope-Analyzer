"""
model_catalog.py - local model profiles for Scope Studio.

The app treats the LLM as a plot/action assistant, not as the numerical
engine. Small models can route actions and explain deterministic tool output;
NumPy/SciPy still compute the numbers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    tier: str
    backend: str
    model: str
    role: str
    notes: str


MODEL_PROFILES = [
    ModelProfile(
        name="Mac direct MLX - Llama 3.2 3B 4-bit",
        tier="balanced",
        backend="mlx",
        model="mlx-community/Llama-3.2-3B-Instruct-4bit",
        role="Fast direct Apple Silicon chat + plot assistant",
        notes=(
            "Preferred Mac-first backend when mlx-lm is installed. Runs "
            "in-process with MLX and keeps Ollama/llama.cpp as fallbacks."
        ),
    ),
    ModelProfile(
        name="Tiny router - FunctionGemma 270M",
        tier="lightweight",
        backend="ollama",
        model="functiongemma",
        role="Fast plot/tool actions",
        notes=(
            "Smallest practical action-router profile. Best for structured "
            "plot commands, not long scientific discussion."
        ),
    ),
    ModelProfile(
        name="Light chat - Llama 3.2 1B",
        tier="lightweight",
        backend="ollama",
        model="llama3.2:1b",
        role="Basic explanations and plot edits",
        notes=(
            "Good first open-source default for students on laptops. Keeps "
            "latency and download size low."
        ),
    ),
    ModelProfile(
        name="Balanced chat/action - Llama 3.2 3B",
        tier="balanced",
        backend="ollama",
        model="llama3.2",
        role="Better instruction following and tool use",
        notes=(
            "Recommended default when Ollama is available and the machine has "
            "a few GB of memory to spare."
        ),
    ),
    ModelProfile(
        name="Apple fast light - Qwen3.5 4B MLX",
        tier="balanced",
        backend="ollama",
        model="qwen3.5:4b-mlx",
        role="Fast chat + routing on Apple Silicon",
        notes=(
            "MLX runner (Ollama >= 0.19) - markedly faster decode than the "
            "GGUF/llama.cpp path on M-series. 4.0 GB resident; text-only "
            "(MLX tags have no image input). ollama pull qwen3.5:4b-mlx"
        ),
    ),
    ModelProfile(
        name="Apple heavy - Qwen3.5 9B MLX",
        tier="heavyweight",
        backend="ollama",
        model="qwen3.5:9b-mlx",
        role="Interpretation tier: anomalies, papers, physics discussion",
        notes=(
            "Recommended heavy tier for M4 Pro 24 GB: 8.9 GB resident, MLX "
            "decode speed, 256K context. Text-only. "
            "ollama pull qwen3.5:9b-mlx"
        ),
    ),
    ModelProfile(
        name="Mac heavy fallback - Qwen 7B GGUF Q4",
        tier="heavyweight",
        backend="llama.cpp",
        model=os.path.expanduser("~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
        role="Richer analysis when Ollama is unavailable",
        notes=(
            "In-process llama.cpp/Metal fallback (needs llama-cpp-python). "
            "Cross-platform GGUF path for non-Mac contributors."
        ),
    ),
]


def profile_names() -> list[str]:
    return [profile.name for profile in MODEL_PROFILES]


def profile_by_name(name: str) -> ModelProfile:
    for profile in MODEL_PROFILES:
        if profile.name == name:
            return profile
    return MODEL_PROFILES[1]


def is_lite() -> bool:
    """True when the SCOPE_STUDIO_LITE env var is set truthy.

    The "Lite" desktop build (and students on small laptops) set this so
    the AI side-chat defaults to a small, low-footprint local model rather
    than the heavyweight M4-Pro default. It only changes the *default*
    model selection — the user can still pick any installed model.
    """
    return os.environ.get("SCOPE_STUDIO_LITE", "").strip().lower() in (
        "1", "true", "yes", "on")


def profiles_by_tier(tier: str) -> list[ModelProfile]:
    return [p for p in MODEL_PROFILES if p.tier == tier]


def default_profile(lite: bool | None = None) -> ModelProfile:
    """Pick a sensible default chat profile.

    lite=True  -> a lightweight-tier chat model (small download/RAM).
    lite=False -> a balanced-tier chat model.
    lite=None  -> decide from :func:`is_lite` (the SCOPE_STUDIO_LITE env).

    The tiny action-router profile is skipped in favor of a chat-capable
    one in the chosen tier.
    """
    if lite is None:
        lite = is_lite()
    tier = "lightweight" if lite else "balanced"
    candidates = profiles_by_tier(tier)
    # Prefer a chat/explanation model; skip tiny action-only routers
    # (their own notes say they're not for scientific discussion).
    for p in candidates:
        role = p.role.lower()
        if "router" not in role and "action" not in role:
            return p
    return candidates[0] if candidates else MODEL_PROFILES[1]
