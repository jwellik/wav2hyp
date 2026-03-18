import pandas as pd
import pytest

from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Magnitude, Origin, OriginQuality, Pick, ResourceIdentifier

from wav2hyp.utils.io import NLLOutput


def _build_minimal_catalog(event_id: str = "event-1", mag: float = 2.5, mag_type: str = "Mw"):
    origin_time = UTCDateTime("2004-09-23T12:00:00")
    quality = OriginQuality(standard_error=0.3, azimuthal_gap=150.0)
    origin = Origin(
        time=origin_time,
        latitude=46.2,
        longitude=-122.18,
        depth=15000.0,  # meters
        quality=quality,
    )
    origin.x = 1.1
    origin.y = 2.2
    origin.z = 3.3

    magnitude = Magnitude(mag=mag, magnitude_type=mag_type)
    picks = [
        Pick(time=origin_time + 10, phase_hint="P"),
        Pick(time=origin_time + 20, phase_hint="S"),
    ]

    event = Event(
        resource_id=ResourceIdentifier(event_id),
        origins=[origin],
        magnitudes=[magnitude],
        picks=picks,
    )
    return Catalog(events=[event])


def test_nll_catalog_table_writer_creates_indexed_table(tmp_path):
    cat = _build_minimal_catalog()
    df = NLLOutput._catalog_to_dataframe(cat)

    out_path = tmp_path / "nll.h5"
    NLLOutput._write_catalog_table(str(out_path), df)

    loaded = pd.read_hdf(out_path, key="catalog_table")
    required = set(NLLOutput.CATALOG_TABLE_COLUMNS)
    assert required.issubset(set(loaded.columns))

    with pd.HDFStore(out_path, mode="r") as store:
        st = store.get_storer("catalog_table")
        table = st.table
        indexed = set(table.colindexes.keys())
    assert set(NLLOutput.CATALOG_TABLE_INDEXED_COLUMNS).issubset(indexed)

    origin_time_iso = pd.Timestamp(loaded["origin_time"].iloc[0]).isoformat()
    where = f"(origin_time >= '{origin_time_iso}') & (origin_time <= '{origin_time_iso}')"
    subset = pd.read_hdf(out_path, key="catalog_table", where=where)
    assert len(subset) == 1


def test_nll_write_writes_tables_with_expected_columns(tmp_path):
    """NLLOutput.write() writes catalog_table (and arrivals_table when catalog has arrivals) with creation_info, comments."""
    cat = _build_minimal_catalog()
    out_path = tmp_path / "nll.h5"
    nll = NLLOutput(str(out_path))
    nll.write(cat, {"config": "test"})

    df_cat = pd.read_hdf(out_path, key="catalog_table")
    assert set(NLLOutput.CATALOG_TABLE_COLUMNS).issubset(set(df_cat.columns))
    assert "creation_info" in df_cat.columns and "comments" in df_cat.columns
    assert len(df_cat) == 1

    with pd.HDFStore(out_path, mode="r") as store:
        if "arrivals_table" in store:
            df_arr = pd.read_hdf(out_path, key="arrivals_table")
            assert set(NLLOutput.ARRIVALS_TABLE_COLUMNS).issubset(set(df_arr.columns))
            assert "creation_info" in df_arr.columns and "comments" in df_arr.columns


def test_nll_read_and_read_catalog_table(tmp_path):
    """read_catalog_table(t1, t2) returns filtered DataFrame; read() requires vdapseisutils."""
    cat = _build_minimal_catalog(event_id="ev1")
    out_path = tmp_path / "nll.h5"
    nll = NLLOutput(str(out_path))
    nll.write(cat, {"config": "test"})

    t1 = UTCDateTime("2004-09-23T11:00:00")
    t2 = UTCDateTime("2004-09-23T13:00:00")

    df = nll.read_catalog_table(t1=t1, t2=t2)
    assert len(df) == 1
    assert str(df["event_id"].iloc[0]).endswith("ev1")

    df_empty = nll.read_catalog_table(t1=UTCDateTime("2000-01-01"), t2=UTCDateTime("2000-01-02"))
    assert len(df_empty) == 0

    try:
        import vdapseisutils  # noqa: F401
    except ImportError:
        return
    vcatalog, metadata = nll.read(t1=t1, t2=t2)
    assert vcatalog is not None
    assert len(vcatalog) == 1
    assert metadata is not None


def test_nll_read_include_arrivals(tmp_path):
    """read(include_arrivals=True) attaches Picks/Arrivals from arrivals_table (requires vdapseisutils)."""
    pytest.importorskip("vdapseisutils")

    from obspy.core.event import Arrival, WaveformStreamID

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
    pick = Pick(
        time=origin_time + 10,
        phase_hint="P",
        waveform_id=waveform_id,
        resource_id=ResourceIdentifier("smi:local/pick-1"),
    )
    arrival = Arrival(phase="P", pick_id=pick.resource_id, time_residual=0.03, time_weight=1.0)
    origin.arrivals = [arrival]
    event = Event(
        resource_id=ResourceIdentifier("event-1"),
        origins=[origin],
        magnitudes=[Magnitude(mag=2.5, magnitude_type="Mw")],
        picks=[pick],
    )
    cat = Catalog(events=[event])

    out_path = tmp_path / "nll.h5"
    nll = NLLOutput(str(out_path))
    nll.write(cat, {"config": "test"})

    vcatalog, _ = nll.read(include_arrivals=True)
    assert len(vcatalog) == 1
    ev = vcatalog[0]
    origin = ev.preferred_origin() or ev.origins[0]
    assert len(getattr(origin, "arrivals", None) or []) >= 1
    assert len(getattr(ev, "picks", None) or []) >= 1
