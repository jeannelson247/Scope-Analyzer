#!/usr/bin/env python3
"""Sync only benchmark-selected shipping MLX models to the model vault.

This script deliberately reads config/shipping_models.json instead of copying
every folder under ~/models/mlx. It is safe to run as a dry run first:

    python3 scripts/sync_shipping_models.py --dry-run

Destination priority when --dest is omitted:
  1. SCOPE_STUDIO_EXTERNAL_MLX_MODELS, if set and on a mounted volume.
  2. /Volumes/<drive>/ScopeStudioModels/mlx
  3. /Volumes/<drive>/Scope Studio Models/mlx
  4. /Volumes/<drive>/models/mlx
  5. /Volumes/<drive>/mlx
  6. ~/models/mlx fallback when no model drive is mounted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "shipping_models.json"
FAT32_SINGLE_FILE_LIMIT_BYTES = (4 * 1024 * 1024 * 1024) - 1


def mounted_mlx_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("SCOPE_STUDIO_EXTERNAL_MLX_MODELS", "").strip()
    if env:
        path = Path(env).expanduser()
        if path.exists():
            roots.append(path)
    volumes = Path("/Volumes")
    if volumes.is_dir():
        for volume in sorted(volumes.iterdir(), key=lambda p: p.name):
            if volume.name in {"Macintosh HD", "Codex Installer"}:
                continue
            for rel in (
                "ScopeStudioModels/mlx",
                "Scope Studio Models/mlx",
                "Scope Studio Models",
                "models/mlx",
                "mlx",
            ):
                path = volume / rel
                if path.is_dir():
                    roots.append(path)
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve())
        if key not in seen:
            out.append(root)
            seen.add(key)
    return out


def default_destination() -> Path:
    roots = mounted_mlx_roots()
    if roots:
        return roots[0]
    return Path.home() / "models" / "mlx"


def source_roots(extra: list[str]) -> list[Path]:
    candidates = [
        Path.home() / "models" / "mlx",
        Path.home() / "models",
        *mounted_mlx_roots(),
    ]
    for item in extra:
        root = Path(item).expanduser()
        candidates.append(root)
        candidates.extend(root / rel for rel in (
            "mlx",
            "ScopeStudioModels/mlx",
            "Scope Studio Models/mlx",
            "models/mlx",
        ))
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def looks_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        names = {p.name for p in path.iterdir()}
    except OSError:
        return False
    return (
        "config.json" in names
        and any(name.startswith("tokenizer") or name in {"vocab.json", "merges.txt"}
                for name in names)
        and any(name.endswith((".safetensors", ".npz")) for name in names)
        and not any(name.endswith(".incomplete") for name in names)
    )


def find_source(name: str, roots: list[Path], dest: Path) -> Path | None:
    for root in roots:
        for candidate in (root / name, root):
            if looks_complete(candidate) and candidate.name == name:
                return candidate
    return None


def load_manifest(include_optional: bool) -> list[dict]:
    data = json.loads(MANIFEST.read_text())
    models = []
    for item in data.get("models", []):
        if not item.get("ship"):
            continue
        if item.get("required") or include_optional:
            models.append(item)
    return models


def copy_model(src: Path, dest: Path, dry_run: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"DRY RUN copy {src} -> {dest}")
        return
    if shutil.which("rsync"):
        subprocess.run(
            ["rsync", "-a", "--delete", f"{src}/", f"{dest}/"],
            check=True,
        )
    else:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)


def human_bytes(size: int | None) -> str:
    if size is None:
        return "unknown size"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def is_fat32_info(text: str) -> bool:
    normalized = text.lower()
    return (
        "dos_fat_32" in normalized
        or "ms-dos fat32" in normalized
        or ("msdos" in normalized and "fat32" in normalized)
    )


def diskutil_probe_path(path: Path) -> Path:
    expanded = path.expanduser()
    parts = expanded.parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return Path(parts[0]) / parts[1] / parts[2]
    probe = expanded
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def filesystem_info(path: Path) -> str:
    """Return best-effort filesystem information for the destination path."""
    probe = diskutil_probe_path(path)
    if sys.platform == "darwin" and shutil.which("diskutil"):
        try:
            result = subprocess.run(
                ["diskutil", "info", str(probe)],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return ""
        return f"{result.stdout}\n{result.stderr}"
    return ""


def largest_remote_file(repo: str) -> tuple[str, int | None]:
    try:
        from huggingface_hub import model_info
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --download-missing. Install it "
            "with `python3 -m pip install -U huggingface_hub`."
        ) from exc
    info = model_info(repo, files_metadata=True)
    largest_name = ""
    largest_size: int | None = None
    for sibling in info.siblings:
        size = getattr(sibling, "size", None)
        if size is None:
            continue
        if largest_size is None or size > largest_size:
            largest_size = int(size)
            largest_name = getattr(sibling, "rfilename", "") or "<unnamed>"
    return largest_name, largest_size


def verify_destination_can_store_repo(repo: str, dest: Path) -> None:
    """Fail before downloading when the target filesystem cannot store a shard."""
    fs_info = filesystem_info(dest.parent)
    if not is_fat32_info(fs_info):
        return
    largest_name, largest_size = largest_remote_file(repo)
    if largest_size is None or largest_size <= FAT32_SINGLE_FILE_LIMIT_BYTES:
        return
    raise RuntimeError(
        f"{repo} contains {largest_name} ({human_bytes(largest_size)}), which "
        "is too large for the FAT32 destination volume. Use an APFS/exFAT "
        "model vault or run without --include-optional. Existing partial "
        "downloads can be deleted safely; Hugging Face downloads resume when "
        "rerun on a compatible filesystem."
    )


def download_model(repo: str, dest: Path, dry_run: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"DRY RUN download {repo} -> {dest}")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --download-missing. Install it "
            "with `python3 -m pip install -U huggingface_hub`."
        ) from exc
    verify_destination_can_store_repo(repo, dest)
    snapshot_download(
        repo_id=repo,
        local_dir=str(dest),
        max_workers=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="", help="Destination mlx folder.")
    parser.add_argument("--source-root", action="append", default=[],
                        help="Additional source root to search.")
    parser.add_argument("--include-optional", action="store_true",
                        help="Also sync optional Pro analyst models.")
    parser.add_argument("--download-missing", action="store_true",
                        help="Download missing selected models with hf.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show planned copies/downloads without writing.")
    args = parser.parse_args()

    dest_root = Path(args.dest).expanduser() if args.dest else default_destination()
    roots = source_roots(args.source_root)
    models = load_manifest(args.include_optional)

    print(f"Destination: {dest_root}")
    print("Selected models:")
    missing = []
    for item in models:
        name = item["name"]
        dst = dest_root / name
        src = find_source(name, roots, dest_root)
        print(f"- {name} [{item['tier']}]")
        if src:
            if src.resolve() == dst.resolve():
                print(f"  already at destination: {dst}")
            else:
                try:
                    copy_model(src, dst, args.dry_run)
                except RuntimeError as exc:
                    print(f"  ERROR: {exc}")
                    return 1
        elif args.download_missing:
            try:
                download_model(item["hf_repo"], dst, args.dry_run)
            except RuntimeError as exc:
                print(f"  ERROR: {exc}")
                return 1
        else:
            missing.append(item)
            print("  missing locally; re-run with --download-missing or add --source-root")

    if missing:
        print("\nMissing selected models:")
        for item in missing:
            print(f"  hf download {item['hf_repo']} --local-dir {dest_root / item['name']}")
        return 2

    print("\nShipping model sync complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
