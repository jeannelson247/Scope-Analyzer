"""
data_session.py - immutable raw-data session metadata for Scope Studio.

The loaded CSV is treated as a read-only measurement. Scope Studio may create
in-memory transforms, overlays, reconstructions, and exports, but it must not
rewrite the source file. The session hash makes that policy testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import os


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DataSession:
    source_path: str
    source_hash: str
    source_size: int
    loaded_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_path(cls, path: str) -> "DataSession":
        return cls(
            source_path=os.path.abspath(path),
            source_hash=file_sha256(path),
            source_size=os.path.getsize(path),
        )

    @property
    def short_hash(self) -> str:
        return self.source_hash[:12]

    def source_unchanged(self) -> bool:
        try:
            return file_sha256(self.source_path) == self.source_hash
        except OSError:
            return False

    def status_line(self) -> str:
        name = os.path.basename(self.source_path)
        mb = self.source_size / (1024 * 1024)
        return (
            f"Original CSV untouched: {name} "
            f"({mb:.2f} MB, sha256 {self.short_hash}...)"
        )
