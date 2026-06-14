from __future__ import annotations

import json

from csv_loader import load_csv
from data_quality import quality_report
from data_session import DataSession
from shot_metadata import ensure_sidecar, sidecar_path_for


def test_ensure_sidecar_writes_provenance_without_waveforms(tek_csv) -> None:
    data = load_csv(str(tek_csv))
    session = DataSession.from_path(str(tek_csv))
    quality = quality_report(data)

    path, meta = ensure_sidecar(data, session, quality)

    assert path == sidecar_path_for(str(tek_csv))
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == meta
    assert saved["schema_version"] == 1
    assert saved["source_sha256"] == session.source_hash
    assert saved["shot_id"] == "T0000"
    assert saved["scope_model"] == "DPO2024B"
    assert saved["n_rows"] == 200
    assert saved["columns"] == ["TIME", "CH1", "CH2"]
    assert saved["quality_status"] == "ok"
    assert "waveform" not in saved
    assert "samples" not in saved


def test_ensure_sidecar_preserves_human_fields(tek_csv) -> None:
    data = load_csv(str(tek_csv))
    session = DataSession.from_path(str(tek_csv))
    quality = quality_report(data)
    sidecar = sidecar_path_for(str(tek_csv))
    sidecar.write_text(
        json.dumps(
            {
                "charging_voltage": 820.0,
                "module_config": "S1-S4 parallel",
                "operator": "Jean",
                "notes": "manual lab context",
            }
        ),
        encoding="utf-8",
    )

    _path, meta = ensure_sidecar(data, session, quality)

    assert meta["charging_voltage"] == 820.0
    assert meta["module_config"] == "S1-S4 parallel"
    assert meta["operator"] == "Jean"
    assert meta["notes"] == "manual lab context"
    assert meta["source_sha256"] == session.source_hash

