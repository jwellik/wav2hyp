"""Tests for overwrite/skip behavior: date_already_processed_for_stage, drop_summary_txt_rows_in_range, remove_range."""

import pandas as pd
import pytest
from obspy import UTCDateTime

from wav2hyp.utils.summary import (
    date_already_processed_for_stage,
    drop_summary_txt_rows_in_range,
    station_summary_reset_for_overwrite,
    station_summary_stage_slice_h5_key,
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
    """When HDF5 stage files do not exist, returns 0 removed slices."""
    result = station_summary_reset_for_overwrite(
        "/nonexistent/picker.h5",
        "/nonexistent/associator.h5",
        "/nonexistent/locator.h5",
        {"locator"},
        "2025-03-19",
        "2025-03-20",
        station_summary_period_seconds=86400,
    )
    assert result["keys_removed"] == 0
    assert result["by_stage"].get("locator") == 0


def test_station_summary_reset_for_overwrite_locator_zeros_nevents(tmp_path):
    """When only locator in cleanup_stages, locator slice node is removed for that date range."""
    picker_h5 = tmp_path / "picker.h5"
    associator_h5 = tmp_path / "associator.h5"
    locator_h5 = tmp_path / "locator.h5"

    # Create one locator slice node for '2025/03/19'
    date_str = "2025/03/19"
    key = station_summary_stage_slice_h5_key("locator", date_str)
    df = pd.DataFrame([{"date": date_str, "trace_id": "A.B.C.D", "nevents": 5}])
    df.to_hdf(str(locator_h5), key=key, mode="w", format="fixed")

    result = station_summary_reset_for_overwrite(
        str(picker_h5),
        str(associator_h5),
        str(locator_h5),
        {"locator"},
        "2025-03-19",
        "2025-03-20",
        station_summary_period_seconds=86400,
    )
    assert result["keys_removed"] == 1
    assert result["by_stage"].get("locator") == 1

    with pd.HDFStore(str(locator_h5), mode="r") as store:
        assert key not in store.keys()


def test_eqt_remove_range_nonexistent(tmp_path):
    """EQTOutput.remove_range on nonexistent file returns zero counts."""
    out_path = tmp_path / "picks.h5"
    eqt = EQTOutput(str(out_path))
    counts = eqt.remove_range("2025-03-19", "2025-03-20")
    assert counts["picks_removed"] == 0 and counts["detections_removed"] == 0


def test_cleanup_stages_for_overwrite_sets():
    """Cascade stage sets match docs: picker clears all; associator clears assoc+loc."""
    pytest.importorskip("vdapseisutils")
    from wav2hyp.core import WAV2HYP

    assert WAV2HYP._cleanup_stages_for_overwrite(True, False, False) == frozenset(
        {"picker", "associator", "locator"}
    )
    assert WAV2HYP._cleanup_stages_for_overwrite(False, True, False) == frozenset(
        {"associator", "locator"}
    )
    assert WAV2HYP._cleanup_stages_for_overwrite(False, False, True) == frozenset({"locator"})
    assert WAV2HYP._cleanup_stages_for_overwrite(False, False, False) == frozenset()


def test_clear_is_associated_in_range_clears_flags(tmp_path):
    """clear_is_associated_in_range sets is_associated to False for picks in the window."""
    t0 = UTCDateTime("2025-03-19T12:00:00")
    t_out = UTCDateTime("2025-03-18T12:00:00")
    h5 = tmp_path / "picks.h5"
    df_in = pd.DataFrame(
        {
            "trace_id": ["NET.STA..BH1", "NET.STA..BH2"],
            "start_time": [t0.datetime, t0.datetime],
            "end_time": [t0.datetime, t0.datetime],
            "peak_time": [t0.datetime, t_out.datetime],
            "peak_value": [0.5, 0.4],
            "phase": ["P", "P"],
            "is_associated": [True, True],
        }
    )
    df_in.to_hdf(
        h5,
        key="picks",
        mode="w",
        format="table",
        data_columns=["peak_time", "is_associated", "peak_value"],
    )
    eqt = EQTOutput(str(h5))
    out = eqt.clear_is_associated_in_range("2025-03-19", "2025-03-19T23:59:59")
    assert out["picks_cleared"] == 1
    df2 = pd.read_hdf(h5, key="picks")
    in_win = df2[df2["trace_id"] == "NET.STA..BH1"]
    out_win = df2[df2["trace_id"] == "NET.STA..BH2"]
    assert bool(in_win["is_associated"].iloc[0]) is False
    assert bool(out_win["is_associated"].iloc[0]) is True
