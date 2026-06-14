from pathlib import Path

import pytest

from ai_assistant import list_mlx_models, resolve_mlx_model, is_hf_model_id


def make_mlx_model(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    (path / "weights.safetensors").write_bytes(b"not real weights")
    return path


def test_hf_model_ids_are_allowed():
    assert is_hf_model_id("mlx-community/Qwen3.5-4B-MLX-4bit")
    assert resolve_mlx_model(
        "mlx-community/Qwen3.5-4B-MLX-4bit"
    ) == "mlx-community/Qwen3.5-4B-MLX-4bit"


def test_parent_folder_resolves_to_complete_mlx_child(tmp_path):
    model = make_mlx_model(tmp_path / "Qwen3.5-4B-MLX-4bit")

    assert list_mlx_models([str(tmp_path)]) == [str(model)]
    assert resolve_mlx_model(str(tmp_path)) == str(model)


def test_incomplete_parent_folder_gets_friendly_error(tmp_path):
    (tmp_path / "not_a_model").mkdir()

    with pytest.raises(ValueError, match="not a complete MLX model folder"):
        resolve_mlx_model(str(tmp_path))


def test_gguf_file_is_rejected_for_direct_mlx(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")

    with pytest.raises(ValueError, match="llama.cpp backend"):
        resolve_mlx_model(str(gguf))
