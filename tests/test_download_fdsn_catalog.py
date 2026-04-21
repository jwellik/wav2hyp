from pathlib import Path

from obspy import UTCDateTime
from obspy.core.event import Arrival, Catalog, Event, Origin, Pick, ResourceIdentifier

from wav2hyp.tools import download_fdsn_catalog as mod


def _build_catalog() -> Catalog:
    origin_time = UTCDateTime("2004-09-23T12:00:00")
    origin = Origin(time=origin_time, latitude=46.2, longitude=-122.18, depth=15000.0)
    pick = Pick(time=origin_time + 10, phase_hint="P", resource_id=ResourceIdentifier("smi:local/p1"))
    origin.arrivals = [Arrival(phase="P", pick_id=pick.resource_id, time_residual=0.01, time_weight=1.0)]
    event = Event(resource_id=ResourceIdentifier("event-1"), origins=[origin], picks=[pick])
    return Catalog(events=[event])


class _FakeClient:
    init_source = None
    init_kwargs = None
    last_query = None

    def __init__(self, source, **kwargs):
        _FakeClient.init_source = source
        _FakeClient.init_kwargs = kwargs

    def get_events(self, **kwargs):
        _FakeClient.last_query = kwargs
        return _build_catalog()


def test_download_fdsn_catalog_endpoint_override_and_arrivals(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "Client", _FakeClient)

    out = tmp_path / "catalog.xml"
    out_path, catalog, comcat_failures = mod.download_fdsn_catalog(
        output_quakeml=str(out),
        datasource="IRIS",
        fdsn_endpoint="1.2.3.4:8080",
        starttime="2024-01-01",
        endtime="2024-01-02",
        includearrivals=True,
        includeallorigins=True,
        includeallmagnitudes=True,
    )

    assert out_path == Path(out)
    assert comcat_failures == []
    assert out.exists()
    assert len(catalog) == 1
    assert _FakeClient.init_source == "http://1.2.3.4:8080"
    assert _FakeClient.last_query["includearrivals"] is True
