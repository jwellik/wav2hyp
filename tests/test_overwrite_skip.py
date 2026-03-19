"""Tests for overwrite/skip behavior: date_already_processed_for_stage, drop_summary_txt_rows_in_range, remove_range."""

import pandas as pd

from wav2hyp.utils.summary import (
    date_already_processed_for_stage,
    drop_summary_txt_rows_in_range,
    station_summary_reset_for_overwrite,
)
from wav2hyp.utils.io import EQTOutput


def test_date_already_processed_for_stage_no_file():
    """When HDF5 does not exist, date_already_processed_for_stage returns False."""
    assert date_already_processed_for_stage("/nonexistent/picker.h5", "2025-03-19", "2025-03-20") is False


def test_date_already_processed_for_stage_no_summary(tmp_path):
    """When HDF5 exists but has no summary table, returns False."""
    h5 = tmp_path / "picker.h5"
    pd.DataFrame({"x": [1]}).to_hdf(h5, key="other", mode="w", format="table")
    assert date_already_processed_for_stage(str(h5), "2025-03-19", "2025-03-20") is False


def test_date_already_processed_for_stage_has_row(tmp_path):
    """When summary has a row for the date range, returns True."""
    h5 = tmp_path / "picker.h5"
    summary = pd.DataFrame([
        {"date": "2025/03/19", "config": "c", "ncha": 1, "t_exec_pick": 1.0, "t_updated_pick": "2025-03-19T00:00:00"},
    ])
    summary.to_hdf(h5, key="summary", mode="w", format="table")
    assert date_already_processed_for_stage(str(h5), "2025-03-19", "2025-03-20") is True


def test_date_already_processed_for_stage_outside_range(tmp_path):
    """When summary has only a row outside the range, returns False."""
    h5 = tmp_path / "picker.h5"
    summary = pd.DataFrame([
        {"date": "2025/03/18", "config": "c", "ncha": 1, "t_exec_pick": 1.0, "t_updated_pick": "2025-03-18T00:00:00"},
    ])
    summary.to_hdf(h5, key="summary", mode="w", format="table")
    assert date_already_processed_for_stage(str(h5), "2025-03-19", "2025-03-20") is False


def test_drop_summary_txt_rows_in_range_nonexistent():
    """When summary .txt does not exist, returns 0."""
    assert drop_summary_txt_rows_in_range("/nonexistent/summary.txt", "2025-03-19", "2025-03-20") == 0


def test_drop_summary_txt_rows_in_range_removes_rows(tmp_path):
    """Rows with date in [t1, t2] are removed and file is rewritten."""
    txt = tmp_path / "summary.txt"
    df = pd.DataFrame([
        {"date": "2025/03/18", "n": 1},
        {"date": "2025/03/19", "n": 2},
        {"date": "2025/03/20", "n": 3},
    ])
    df.to_csv(txt, index=False)
    removed = drop_summary_txt_rows_in_range(str(txt), "2025-03-19", "2025-03-20")
    assert removed == 2
    out = pd.read_csv(txt)
    assert len(out) == 1
    assert out["date"].iloc[0] == "2025/03/18"


def test_station_summary_reset_for_overwrite_no_file():
    """When station summary file does not exist, returns zeros."""
    result = station_summary_reset_for_overwrite(
        "/nonexistent", "station_summary.txt", {"locator"}, "2025-03-19", "2025-03-20"
    )
    assert result["rows_removed"] == 0 and result["rows_reset"] == 0


def test_station_summary_reset_for_overwrite_locator_zeros_nevents(tmp_path):
    """When only locator in cleanup_stages, nevents is zeroed for rows in date range."""
    path = tmp_path / "station_summary.txt"
    df = pd.DataFrame([
        {"date": "2025/03/19", "trace_id": "A.B.C.D", "nevents": 5, "nassign": 3, "nassoc": 2},
    ])
    df.to_csv(path, index=False)
    result = station_summary_reset_for_overwrite(
        str(tmp_path), "station_summary.txt", {"locator"}, "2025-03-19", "2025-03-20"
    )
    assert result["rows_removed"] == 0 and result["rows_reset"] == 1
    out = pd.read_csv(path)
    assert out["nevents"].iloc[0] == 0
    assert out["nassign"].iloc[0] == 3 and out["nassoc"].iloc[0] == 2


def test_eqt_remove_range_nonexistent(tmp_path):
    """EQTOutput.remove_range on nonexistent file returns zero counts."""
    out_path = tmp_path / "picks.h5"
    eqt = EQTOutput(str(out_path))
    counts = eqt.remove_range("2025-03-19", "2025-03-20")
    assert counts["picks_removed"] == 0 and counts["detections_removed"] == 0
