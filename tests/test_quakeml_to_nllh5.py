from pathlib import Path

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

from obspy import read_events

from wav2hyp.tools.quakeml_to_nllh5 import quakeml_to_nllh5
from wav2hyp.utils.io import NLLOutput


def _build_quakeml_with_one_arrival(path: Path) -> None:
    origin_time = UTCDateTime("2004-09-23T12:00:00")
    quality = OriginQuality(standard_error=0.3, azimuthal_gap=150.0)
    origin = Origin(
        time=origin_time,
        latitude=46.2,
        longitude=-122.18,
        depth=15000.0,
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
        resource_id=ResourceIdentifier("event-1"),
        origins=[origin],
        magnitudes=[magnitude],
        picks=[pick],
    )
    cat = Catalog(events=[event])
    cat.write(str(path), format="QUAKEML")


def test_quakeml_to_nllh5_writes_tables(tmp_path):
    quakeml_path = tmp_path / "in.xml"
    _build_quakeml_with_one_arrival(quakeml_path)

    out_path = tmp_path / "nll.h5"
    quakeml_to_nllh5(str(quakeml_path), output_path=str(out_path), verbose=False)

    assert out_path.exists()

    df_cat = pd.read_hdf(out_path, key="catalog_table")
    assert len(df_cat) == 1
    assert "event_id" in df_cat.columns

    df_arr = pd.read_hdf(out_path, key="arrivals_table")
    assert len(df_arr) == 1
    assert df_arr.iloc[0]["phase"] == "P"
    assert df_arr.iloc[0]["station_id"] == "UW.HSR"

    # Smoke test: attach from arrivals_table (avoid NLLOutput.read(),
    # which requires vdapseisutils in some environments).
    catalog = read_events(str(quakeml_path))
    # Clear arrivals to prove attachment comes from arrivals_table.
    ev = catalog[0]
    origin = ev.preferred_origin() if hasattr(ev, "preferred_origin") else None
    if origin is None:
        origin = ev.origins[0]
    origin.arrivals = []

    nll = NLLOutput(str(out_path))
    arrivals_df = nll.read_arrivals(event_ids=[str(df_arr.iloc[0]["event_id"])])
    NLLOutput.attach_arrivals_to_catalog(catalog, arrivals_df)
    assert len(origin.arrivals) == 1

