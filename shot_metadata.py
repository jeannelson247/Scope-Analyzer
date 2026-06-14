"""Structured sidecar metadata for experimental shots.

Sidecars are intentionally small JSON files next to the source CSV:
`<shot>.meta.json`. They contain provenance and lab context, never waveform
samples. Existing human-entered fields are preserved when the loader refreshes
derived fields such as hash, row count, and scope model.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HUMAN_FIELDS = {
    "charging_voltage",
    "module_config",
    "operator",
    "notes",
}


def sidecar_path_for(source_path: str | os.PathLike[str]) -> Path:
    path = Path(source_path)
    return path.with_suffix(".meta.json")


def build_metadata(data, session, quality=None) -> dict[str, Any]:
    meta = getattr(data, "meta", {}) or {}
    model = first_meta_value(meta.get("Model"))
    sample_interval = (
        getattr(quality, "sample_interval_s", None)
        if quality is not None
        else infer_sample_interval(data)
    )
    source_path = os.path.abspath(getattr(data, "path", session.source_path))
    return {
        "schema_version": 1,
        "source_path": source_path,
        "source_sha256": session.source_hash,
        "source_size": session.source_size,
        "shot_id": Path(source_path).stem,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope_model": model or None,
        "sample_interval_s": sample_interval,
        "n_rows": int(getattr(data, "n_rows", len(data.df))),
        "columns": list(getattr(data, "columns", list(data.df.columns))),
        "units": dict(getattr(data, "units", {}) or {}),
        "quality_status": getattr(quality, "status", None),
        "quality_issues": list(getattr(quality, "issues", []) or []),
        "charging_voltage": None,
        "module_config": None,
        "operator": None,
        "notes": "",
    }


def ensure_sidecar(data, session, quality=None) -> tuple[Path, dict[str, Any]]:
    path = sidecar_path_for(data.path)
    generated = build_metadata(data, session, quality)
    existing = read_json(path)
    if existing:
        for field in HUMAN_FIELDS:
            if existing.get(field) not in (None, ""):
                generated[field] = existing[field]
    path.write_text(json.dumps(generated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, generated


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def first_meta_value(value) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return str(value) if value else ""


def infer_sample_interval(data) -> float | None:
    try:
        import numpy as np

        x = data.df.iloc[:, 0].to_numpy(dtype=float, copy=False)
        diffs = np.diff(x)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        return float(np.median(diffs)) if diffs.size else None
    except Exception:
        return None
