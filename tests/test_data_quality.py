from __future__ import annotations

import numpy as np
import pandas as pd

from csv_loader import LoadedData, load_csv
from data_quality import quality_report


def test_quality_report_ok_for_clean_scope_file(tek_csv) -> None:
    data = load_csv(str(tek_csv))
    rep = quality_report(data)

    assert rep.status == "ok"
    assert rep.ok is True
    assert rep.n_rows == 200
    assert rep.n_columns == 3
    assert rep.x_column == "TIME"
    assert rep.sample_interval_s is not None
    assert rep.sample_rate_hz is not None
    assert rep.total_nonfinite_values == 0
    assert "QC OK" in rep.one_line()


def test_quality_report_flags_bad_timing_and_missing_data() -> None:
    df = pd.DataFrame(
        {
            "TIME": [0.0, 1.0, 1.0, 0.5, 10.0, np.nan],
            "CH1": [1.0, np.nan, 2.0, 3.0, 4.0, 5.0],
            "CH2": [1.0, 2.0, np.inf, 4.0, 5.0, 6.0],
        }
    )
    data = LoadedData(
        path="bad.csv",
        df=df,
        delimiter=",",
        skiprows=0,
        columns=list(df.columns),
    )

    rep = quality_report(data)

    assert rep.status == "error"
    assert rep.nonfinite_time_count == 1
    assert rep.duplicate_timestamp_count == 1
    assert rep.backwards_timestamp_count == 1
    assert rep.total_nonfinite_values == 3
    assert rep.nonfinite_by_column["TIME"] == 1
    assert rep.nonfinite_by_column["CH1"] == 1
    assert rep.nonfinite_by_column["CH2"] == 1
    assert "QC ERROR" in rep.one_line()


def test_quality_report_flags_large_time_gap_as_warning() -> None:
    df = pd.DataFrame(
        {
            "TIME": [0.0, 1.0, 2.0, 3.0, 100.0],
            "CH1": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    data = LoadedData(
        path="gap.csv",
        df=df,
        delimiter=",",
        skiprows=0,
        columns=list(df.columns),
    )

    rep = quality_report(data)

    assert rep.status == "warning"
    assert rep.large_gap_count == 1
    assert rep.max_gap_s == 97.0
    assert "large timestamp gap" in rep.issues[0]


def test_quality_report_flags_flatline_dropout_candidate() -> None:
    t = np.linspace(0.0, 0.100, 1000)
    y = np.sin(2 * np.pi * 60 * t)
    y[300:420] = 0.0
    df = pd.DataFrame({"TIME": t, "CH1": y})
    data = LoadedData(
        path="flatline.csv",
        df=df,
        delimiter=",",
        skiprows=0,
        columns=list(df.columns),
    )

    rep = quality_report(data)

    assert rep.status == "warning"
    assert rep.flatline_runs_by_column["CH1"] == 1
    assert rep.longest_flatline_by_column["CH1"]["samples"] == 120
    assert "flatline/dropout" in rep.text()
