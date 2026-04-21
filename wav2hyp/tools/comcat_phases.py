"""
USGS ComCat phase-data (QuakeML) via usgs-libcomcat.

Fetches the ``phase-data`` product for each ObsPy :class:`~obspy.core.event.Event`
using the public id expected by ComCat (from ``resource_id`` / ``eventid`` query),
then replaces ``event.picks`` with picks parsed from that product's ``quakeml.xml``.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Optional

import requests
from obspy import Catalog
from obspy.io.quakeml.core import Unpickler

from wav2hyp.tools.fdsn_detail_catalog import fdsn_eventid_for_get_events

_log = logging.getLogger("wav2hyp")


def _phase_data_quakeml_catalog(detail, source: str = "preferred") -> Optional[Catalog]:
    """Load phase-data ``quakeml.xml`` into an ObsPy :class:`~obspy.core.event.Catalog`."""
    from libcomcat.utils import HEADERS, TIMEOUT

    if source is None:
        source = "preferred"
    if not detail.hasProduct("phase-data"):
        return None
    phasedata = detail.getProducts("phase-data", source=source)[0]
    quakeurl = phasedata.getContentURL("quakeml.xml")
    try:
        response = requests.get(quakeurl, timeout=TIMEOUT, headers=HEADERS)
        data = response.text.encode("utf-8")
    except Exception as exc:
        _log.debug("ComCat phase-data GET failed: %s", exc)
        return None
    unpickler = Unpickler()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            return unpickler.loads(data)
        except Exception as exc:
            _log.debug("ComCat phase QuakeML parse failed: %s", exc)
            return None


def enrich_catalog_with_comcat_phases(
    catalog: Catalog,
    *,
    comcat_search_catalog: Optional[str] = None,
    comcat_phase_source: str = "preferred",
    delay_seconds: float = 0.25,
    progress: bool = False,
) -> tuple[Catalog, list[tuple[str, str]]]:
    """
    For each event, resolve a ComCat id from ``resource_id``, fetch phase-data QuakeML,
    and set ``event.picks`` from that product.

    Events with no ComCat match, no ``phase-data`` product, or parse errors are left
    unchanged (picks as returned by FDSN); failures are collected in the second
    return value.

    Parameters
    ----------
    comcat_search_catalog
        Passed to :func:`libcomcat.search.get_event_by_id` as ``catalog`` (ComCat
        catalog filter), or ``None`` for any.
    comcat_phase_source
        Phase product source: ``preferred`` or a network code (``us``, ``uw``, etc.).
    """
    from libcomcat.search import get_event_by_id

    failures: list[tuple[str, str]] = []
    n = len(catalog)
    for i, event in enumerate(catalog):
        rid = getattr(getattr(event, "resource_id", None), "id", None)
        if not rid:
            failures.append(("", "no resource_id on event"))
            continue
        eid = fdsn_eventid_for_get_events(str(rid))
        if not eid:
            failures.append(("", "empty event id from resource_id"))
            continue
        if delay_seconds > 0 and i > 0:
            time.sleep(delay_seconds)
        try:
            detail = get_event_by_id(eid, catalog=comcat_search_catalog)
        except Exception as exc:
            failures.append((eid, str(exc)))
            if progress:
                print(f"[comcat {i + 1}/{n}] eventid={eid} ERROR {exc!s}", flush=True)
            continue
        if not detail.hasProduct("phase-data"):
            failures.append((eid, "no phase-data product"))
            if progress:
                print(f"[comcat {i + 1}/{n}] eventid={eid} no phase-data", flush=True)
            continue
        try:
            phase_cat = _phase_data_quakeml_catalog(detail, source=comcat_phase_source)
        except Exception as exc:
            failures.append((eid, str(exc)))
            if progress:
                print(f"[comcat {i + 1}/{n}] eventid={eid} product ERROR {exc!s}", flush=True)
            continue
        if phase_cat is None or len(phase_cat.events) == 0:
            failures.append((eid, "empty or unreadable phase quakeml"))
            if progress:
                print(f"[comcat {i + 1}/{n}] eventid={eid} empty phase quakeml", flush=True)
            continue
        pe = phase_cat[0]
        event.picks = list(getattr(pe, "picks", None) or [])
        np = len(event.picks)
        if progress:
            print(f"[comcat {i + 1}/{n}] eventid={eid} picks={np}", flush=True)
    return catalog, failures
