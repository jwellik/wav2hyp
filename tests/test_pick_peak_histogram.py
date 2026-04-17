import pandas as pd

from wav2hyp.utils.io import EQTOutput
from wav2hyp.utils.stations import station_from_trace_id


def test_station_from_trace_id_basic():
    assert station_from_trace_id("NET.STA.LOC.CHAN") == "NET.STA"
    assert station_from_trace_id("NET.STA") == "NET.STA"
    assert station_from_trace_id("STA") == "STA"


def test_pick_peak_histogram_per_station(tmp_path, monkeypatch):
    """
    Basic behavioural test: ensure that pick_peak_histogram rows are written per station
    and that aggregating over station_id reproduces global counts.
    """
    # Minimal fake EQTOutput with in-memory picks/detections and summary_stats
    output_path = tmp_path / "eqt-volpick.h5"
    eqt = EQTOutput(str(output_path))

    # Build small picks/detections DataFrames
    picks_df = pd.DataFrame(
        {
            "trace_id": ["NET.AAA..BHZ", "NET.BBB..BHZ"],
            "peak_time": pd.to_datetime(
                ["2000-01-01T00:00:00", "2000-01-01T00:00:10"]
            ),
            "peak_value": [0.5, 0.7],
            "phase": ["P", "S"],
        }
    )
    detections_df = pd.DataFrame(
        {
            "trace_id": ["NET.AAA..BHZ", "NET.BBB..BHZ"],
            "start_time": pd.to_datetime(
                ["2000-01-01T00:00:00", "2000-01-01T00:00:10"]
            ),
            "end_time": pd.to_datetime(
                ["2000-01-01T00:00:05", "2000-01-01T00:00:15"]
            ),
            "peak_value": [0.6, 0.8],
        }
    )

    # Monkeypatch internals so we can call the private writer with our DataFrames
    summary_stats = {
        "date": "2000/01/01",
        "config": "test",
        "ncha": 2,
        "nsamp": 0,
        "pick_model": "test",
        "np": 1,
        "ns": 1,
        "npicks": 2,
        "ndetections": 2,
        "p_thresh": 0.0,
        "s_thresh": 0.0,
        "d_thresh": 0.0,
        "t_exec_pick": 0.0,
    }

    # Write minimal picks/detections tables so _write_picker_summary_and_histogram
    # can append the histogram to the same file.
    picks_df.to_hdf(output_path, key="picks", mode="w", format="table")
    detections_df.to_hdf(output_path, key="detections", mode="a", format="table")

    eqt._write_picker_summary_and_histogram(  # type: ignore[attr-defined]
        summary_stats,
        picks_df=picks_df,
        detections_df=detections_df,
        existing_summary_df=None,
        existing_hist_df=None,
    )

    hist = pd.read_hdf(output_path, key="pick_peak_histogram")

    # Expect per-station rows
    assert {"NET.AAA", "NET.BBB"} <= set(hist["station_id"].unique())

    # Aggregated over station_id, counts per phase/peak_value_bin should match 1
    agg = (
        hist.groupby(["phase", "peak_value_bin"])["count"]
        .sum()
        .reset_index()
    )
    assert set(agg["count"].unique()) == {1}

