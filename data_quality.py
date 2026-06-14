"""Headless data-quality checks for loaded scope files.

The UI can display this as a short banner, but the checks live here so they
are reproducible, testable, and usable from scripts/CI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DataQualityReport:
    status: str
    n_rows: int
    n_columns: int
    x_column: str
    sample_interval_s: float | None
    sample_rate_hz: float | None
    nonfinite_time_count: int
    duplicate_timestamp_count: int
    backwards_timestamp_count: int
    large_gap_count: int
    max_gap_s: float | None
    total_nonfinite_values: int
    nonfinite_by_column: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def one_line(self) -> str:
        dt = "unknown dt" if self.sample_interval_s is None else (
            f"dt {self.sample_interval_s:.3g}s"
        )
        sr = "" if self.sample_rate_hz is None else (
            f", {self.sample_rate_hz:.3g} Hz"
        )
        missing = (
            "no NaNs/nonfinite"
            if self.total_nonfinite_values == 0
            else f"{self.total_nonfinite_values} NaN/nonfinite"
        )
        if self.status == "ok":
            return f"QC OK: {self.n_rows:,} rows, {dt}{sr}, {missing}"
        issue = self.issues[0] if self.issues else "review data quality"
        return (
            f"QC {self.status.upper()}: {self.n_rows:,} rows, {dt}{sr}, "
            f"{missing}; {issue}"
        )


def quality_report(data, x_column: str | None = None) -> DataQualityReport:
    """Return a deterministic QC summary for a `csv_loader.LoadedData`.

    Checks intentionally stay conservative: anything that can affect timing or
    numerical analysis becomes an issue; NaNs in non-time columns are warnings.
    """
    df = data.df
    columns = [str(c) for c in df.columns]
    x_column = x_column or (columns[0] if columns else "")
    issues: list[str] = []

    if not columns or len(df) == 0:
        return DataQualityReport(
            status="error",
            n_rows=len(df),
            n_columns=len(columns),
            x_column=x_column,
            sample_interval_s=None,
            sample_rate_hz=None,
            nonfinite_time_count=0,
            duplicate_timestamp_count=0,
            backwards_timestamp_count=0,
            large_gap_count=0,
            max_gap_s=None,
            total_nonfinite_values=0,
            issues=["file contains no numeric table"],
        )

    if x_column not in df.columns:
        x_column = columns[0]

    arr = df.to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(arr)
    nonfinite_by_column = {
        col: int((~finite[:, idx]).sum())
        for idx, col in enumerate(columns)
        if int((~finite[:, idx]).sum()) > 0
    }
    total_nonfinite = int((~finite).sum())

    x = df[x_column].to_numpy(dtype=np.float64, copy=False)
    x_finite = np.isfinite(x)
    nonfinite_time = int((~x_finite).sum())
    if nonfinite_time:
        issues.append(f"{nonfinite_time} nonfinite timestamp(s)")

    xf = x[x_finite]
    diffs = np.diff(xf)
    duplicate_count = int(np.count_nonzero(diffs == 0))
    backwards_count = int(np.count_nonzero(diffs < 0))
    if duplicate_count:
        issues.append(f"{duplicate_count} duplicate timestamp step(s)")
    if backwards_count:
        issues.append(f"{backwards_count} backwards timestamp step(s)")

    positive = diffs[diffs > 0]
    sample_interval = float(np.median(positive)) if positive.size else None
    sample_rate = 1.0 / sample_interval if sample_interval and sample_interval > 0 else None
    large_gap_count = 0
    max_gap = float(np.nanmax(positive)) if positive.size else None
    if sample_interval and positive.size:
        large_gap_count = int(np.count_nonzero(positive > 5.0 * sample_interval))
        if large_gap_count:
            issues.append(f"{large_gap_count} large timestamp gap(s)")

    if total_nonfinite and not nonfinite_time:
        issues.append(f"{total_nonfinite} NaN/nonfinite data value(s)")
    if sample_interval is None:
        issues.append("could not infer a positive sample interval")

    if nonfinite_time or backwards_count or sample_interval is None:
        status = "error"
    elif duplicate_count or large_gap_count or total_nonfinite:
        status = "warning"
    else:
        status = "ok"

    return DataQualityReport(
        status=status,
        n_rows=len(df),
        n_columns=len(columns),
        x_column=x_column,
        sample_interval_s=sample_interval,
        sample_rate_hz=sample_rate,
        nonfinite_time_count=nonfinite_time,
        duplicate_timestamp_count=duplicate_count,
        backwards_timestamp_count=backwards_count,
        large_gap_count=large_gap_count,
        max_gap_s=max_gap,
        total_nonfinite_values=total_nonfinite,
        nonfinite_by_column=nonfinite_by_column,
        issues=issues,
    )

