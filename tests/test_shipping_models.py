import json
from pathlib import Path

from scripts.sync_shipping_models import (
    FAT32_SINGLE_FILE_LIMIT_BYTES,
    diskutil_probe_path,
    find_source,
    human_bytes,
    is_fat32_info,
    load_manifest,
    looks_complete,
)


ROOT = Path(__file__).resolve().parents[1]


def test_shipping_manifest_has_only_benchmark_selected_models():
    manifest = json.loads((ROOT / "config" / "shipping_models.json").read_text())
    names = {item["name"] for item in manifest["models"] if item["ship"]}

    assert names == {
        "Qwen3.5-4B-MLX-4bit",
        "Llama-3.2-3B-Instruct-4bit",
        "Qwen3-14B-4bit",
    }
    assert all(item["benchmark_summary"] for item in manifest["models"])
    assert all(not item["name"].startswith("NVIDIA-Nemotron")
               for item in manifest["models"])


def test_required_shipping_set_excludes_optional_by_default():
    names = {item["name"] for item in load_manifest(include_optional=False)}

    assert names == {
        "Qwen3.5-4B-MLX-4bit",
        "Llama-3.2-3B-Instruct-4bit",
    }


def test_model_completeness_check_requires_weights_tokenizer_and_config(tmp_path):
    incomplete = tmp_path / "Incomplete"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}")
    assert not looks_complete(incomplete)

    complete = tmp_path / "Complete"
    complete.mkdir()
    (complete / "config.json").write_text("{}")
    (complete / "tokenizer.json").write_text("{}")
    (complete / "weights.safetensors").write_bytes(b"x")
    assert looks_complete(complete)


def test_missing_destination_is_not_counted_as_existing_source(tmp_path):
    dest = tmp_path / "vault"
    dest.mkdir()

    assert find_source("Qwen3.5-4B-MLX-4bit", [dest], dest) is None


def test_fat32_detector_catches_macos_diskutil_output():
    text = """
       Partition Type:            DOS_FAT_32
       File System Personality:   MS-DOS FAT32
       Type (Bundle):             msdos
    """

    assert is_fat32_info(text)
    assert FAT32_SINGLE_FILE_LIMIT_BYTES < 4 * 1024 * 1024 * 1024
    assert human_bytes(5 * 1024 * 1024 * 1024) == "5.0 GiB"


def test_diskutil_probe_uses_volume_root_for_nested_model_paths():
    path = Path("/Volumes/JEAN D2/Scope Studio Models/mlx/Qwen3-14B-4bit")

    assert diskutil_probe_path(path) == Path("/Volumes/JEAN D2")
