import pandas as pd
from obspy import UTCDateTime
from obspy.core.event import (
    Arrival,
    Catalog,
    Event,
    Magnitude,
    Origin,
    OriginQuality,
    Pick,
    ResourceIdentifier,
    WaveformStreamID,
)

from wav2hyp.utils.io import NLLOutput


def _build_minimal_catalog_with_arrival(event_id: str = "event-1"):
    origin_time = UTCDateTime("2004-09-23T12:00:00")
    quality = OriginQuality(standard_error=0.3, azimuthal_gap=150.0)
    origin = Origin(
        time=origin_time,
        latitude=46.2,
        longitude=-122.18,
        depth=15000.0,  # meters
        quality=quality,
    )

    waveform_id = WaveformStreamID(seed_string="UW.HSR.--.BHZ")
    pick_time = origin_time + 10
    pick = Pick(
        time=pick_time,
        phase_hint="P",
        waveform_id=waveform_id,
        resource_id=ResourceIdentifier("smi:local/pick-1"),
    )

    arrival = Arrival(
        phase="P",
        pick_id=pick.resource_id,
        time_residual=0.03,
        time_weight=1.0,
    )
    origin.arrivals = [arrival]

    magnitude = Magnitude(mag=2.5, magnitude_type="Mw")
    event = Event(
        resource_id=ResourceIdentifier(event_id),
        origins=[origin],
        magnitudes=[magnitude],
        picks=[pick],
    )
    return Catalog(events=[event]), origin_time, pick_time


def test_arrivals_table_extraction_and_query(tmp_path):
    cat, origin_time, pick_time = _build_minimal_catalog_with_arrival(event_id="event-1")

    df = NLLOutput._catalog_arrivals_to_dataframe(cat)
    assert len(df) == 1
    assert set(NLLOutput.ARRIVALS_TABLE_COLUMNS).issubset(set(df.columns))

    row = df.iloc[0]
    assert row["event_id"] == "event-1"
    assert row["phase"] == "P"
    assert row["trace_id"] == "UW.HSR.--.BHZ"
    assert row["station_id"] == "UW.HSR"
    assert abs(float(row["residual"]) - 0.03) < 1e-9
    assert abs(float(row["weight"]) - 1.0) < 1e-9
    assert pd.notna(row["arrival_time"])

    out_path = tmp_path / "nll.h5"
    NLLOutput._write_arrivals_table(str(out_path), df)

    nll = NLLOutput(str(out_path))
    df2 = nll.read_arrivals(
        event_ids=["event-1"],
        t1=origin_time - 1,
        t2=pick_time + 1,
    )
    assert len(df2) == 1
    assert df2.iloc[0]["event_id"] == "event-1"


def test_attach_arrivals_to_catalog(tmp_path):
    cat, origin_time, pick_time = _build_minimal_catalog_with_arrival(event_id="event-1")

    # Build arrivals_df from the original catalog
    arrivals_df = NLLOutput._catalog_arrivals_to_dataframe(cat)

    # Create a new catalog with an empty origin.arrivals to verify attachment.
    ev = cat[0]
    origin = ev.origins[0]
    origin.arrivals = []
    ev.picks = ev.picks  # keep picks

    NLLOutput.attach_arrivals_to_catalog(cat, arrivals_df)

    origin2 = cat[0].origins[0]
    assert len(origin2.arrivals) == 1
    arr = origin2.arrivals[0]
    assert arr.phase == "P"
    assert abs(float(arr.time_residual) - 0.03) < 1e-9
    assert abs(float(arr.time_weight) - 1.0) < 1e-9
    assert arr.pick_id.id == "smi:local/pick-1"

