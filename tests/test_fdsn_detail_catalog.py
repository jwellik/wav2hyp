"""Tests for bulk-then-detail FDSN catalog fetch."""

from unittest.mock import patch

from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Origin, ResourceIdentifier

from wav2hyp.tools import fdsn_detail_catalog as mod


def _minimal_event(eid: str) -> Event:
    ot = UTCDateTime("2004-09-23T12:00:00")
    origin = Origin(time=ot, latitude=46.2, longitude=-122.18, depth=5000.0)
    return Event(resource_id=ResourceIdentifier(eid), origins=[origin])


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def get_events(self, **kwargs):
        if kwargs.get("eventid"):
            # IRIS expects numeric eventid; ObsPy passes the string we supply.
            return Catalog(events=[_minimal_event(str(kwargs["eventid"]))])
        # bulk: IRIS-style URIs (numeric id in query string); duplicate + one unique
        return Catalog(
            events=[
                _minimal_event("smi:service.iris.edu/fdsnws/event/1/query?eventid=1"),
                _minimal_event("smi:service.iris.edu/fdsnws/event/1/query?eventid=1"),
                _minimal_event("smi:service.iris.edu/fdsnws/event/1/query?eventid=2"),
            ]
        )


def test_fdsn_eventid_extracts_iris_numeric():
    assert (
        mod.fdsn_eventid_for_get_events(
            "smi:service.iris.edu/fdsnws/event/1/query?eventid=1765568"
        )
        == "1765568"
    )
    assert mod.fdsn_eventid_for_get_events("1765568") == "1765568"


def test_fdsn_eventid_extracts_usgs_quakeml_uri():
    uri = (
        "quakeml:earthquake.usgs.gov/fdsnws/event/1/query?"
        "eventid=uw10659298&format=quakeml"
    )
    assert mod.fdsn_eventid_for_get_events(uri) == "uw10659298"


def test_datasource_supports_include_arrivals():
    assert mod.datasource_supports_include_arrivals("USGS") is False
    assert mod.datasource_supports_include_arrivals("IRIS") is True
    assert mod.datasource_supports_include_arrivals("IRIS", "https://earthquake.usgs.gov") is False


def test_event_ids_for_detail_fetch_dedupes_on_normalized():
    cat = Catalog(
        events=[
            _minimal_event("smi:service.iris.edu/fdsnws/event/1/query?eventid=9"),
            _minimal_event("smi:service.iris.edu/fdsnws/event/1/query?eventid=9"),
            _minimal_event("smi:service.iris.edu/fdsnws/event/1/query?eventid=10"),
        ]
    )
    assert mod.event_ids_for_detail_fetch(cat) == ["9", "10"]


def test_event_ids_dedupe_preserves_order():
    cat = Catalog(
        events=[
            _minimal_event("a"),
            _minimal_event("b"),
            _minimal_event("a"),
        ]
    )
    ids = mod.event_ids_from_catalog(cat, dedupe=True)
    assert ids == ["a", "b"]


class _CaptureClient:
    """Records kwargs for each get_events call."""

    def __init__(self):
        self.calls: list[dict] = []

    def get_events(self, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs.get("eventid"):
            return Catalog(events=[_minimal_event(str(kwargs["eventid"]))])
        return Catalog(
            events=[
                _minimal_event("smi:service.iris.edu/fdsnws/event/1/query?eventid=1"),
                _minimal_event("smi:service.iris.edu/fdsnws/event/1/query?eventid=2"),
            ]
        )


def test_fetch_events_detail_omits_includearrivals_when_unsupported():
    cap = _CaptureClient()
    merged, failures = mod.fetch_events_detail_merged(
        cap,
        ["uw10659298"],
        delay_seconds=0,
        supports_include_arrivals=False,
    )
    assert not failures
    assert len(cap.calls) == 1
    assert "includearrivals" not in cap.calls[0]


@patch.object(mod, "Client", _FakeClient)
def test_download_bulk_then_detail_merges_and_rate_limits(tmp_path):
    out = tmp_path / "out.xml"
    with patch("wav2hyp.tools.fdsn_detail_catalog.time.sleep") as sl:
        path, merged, failures, comcat_failures = mod.download_fdsn_catalog_bulk_then_detail(
            str(out),
            datasource="IRIS",
            starttime="2004-09-23",
            endtime="2004-10-15",
            delay_seconds=0.1,
            bulk_include_arrivals=False,
        )
    assert path == out
    assert comcat_failures == []
    assert out.exists()
    assert len(merged) == 2
    assert failures == []
    # 3 bulk events -> 2 unique ids -> 2 detail calls -> 1 sleep between
    assert sl.call_count == 1
