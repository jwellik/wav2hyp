"""
FDSN: bulk event list, then per-event detail with arrivals.

Many services omit or strip ``Origin.arrivals`` on wide-area ``get_events`` queries.
Fetching each event by ``eventid`` with ``includearrivals=True`` often returns full
QuakeML suitable for wav2hyp ``arrivals_table`` extraction.

This module provides:
- deduplicated event id extraction from a bulk ``Catalog``
- IRIS: normalize ``resource_id`` URIs to numeric ``eventid`` for FDSN queries
- rate-limited per-event ``get_events(eventid=...)`` (``includearrivals`` omitted for USGS — unsupported)
- merged ``obspy.Catalog`` for write / conversion
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

from obspy import Catalog
from obspy.clients.fdsn import Client

from wav2hyp.tools.download_fdsn_catalog import _resolve_datasource

_log = logging.getLogger("wav2hyp")


def _count_picks_and_arrivals(catalog: Catalog) -> tuple[int, int]:
    """Total picks and origin arrivals across all events in *catalog*."""
    n_picks = 0
    n_arrivals = 0
    for ev in catalog:
        n_picks += len(getattr(ev, "picks", None) or [])
        origin = None
        if hasattr(ev, "preferred_origin") and callable(ev.preferred_origin):
            origin = ev.preferred_origin()
        if origin is None:
            origins = getattr(ev, "origins", None) or []
            origin = origins[0] if origins else None
        if origin is not None:
            n_arrivals += len(getattr(origin, "arrivals", None) or [])
    return n_picks, n_arrivals

# IRIS: ``smi:...?eventid=1765568`` — pass ``1765568`` only.
# USGS: ``quakeml:earthquake.usgs.gov/...?eventid=uw10659298&format=quakeml`` — pass ``uw10659298`` only.
_EVENTID_IN_QUERY = re.compile(r"(?:^|[?&])eventid=([^&\s#]+)", re.IGNORECASE)


def datasource_supports_include_arrivals(
    datasource: Optional[str], fdsn_endpoint: Optional[str] = None
) -> bool:
    """
    USGS fdsnws-event returns HTTP 501 if ``includearrivals`` is sent — omit it entirely.
    """
    ep = (fdsn_endpoint or "").strip().lower()
    if "usgs.gov" in ep:
        return False
    ds = (datasource or "").strip().upper()
    if ds == "USGS":
        return False
    return True


def fdsn_eventid_for_get_events(resource_id_str: str) -> str:
    """
    Map ObsPy ``resource_id.id`` to the value FDSN ``get_events(eventid=...)`` expects.

    Extract ``eventid`` from ``...?eventid=VALUE&...`` when present (IRIS numeric id,
    USGS catalog id like ``uw10659298``). Otherwise return a bare id (digits-only or
    simple public id without URI noise).
    """
    s = (resource_id_str or "").strip()
    if not s:
        return s
    s_amp = s.replace("&amp;", "&")
    m = _EVENTID_IN_QUERY.search(s_amp)
    if m:
        return unquote(m.group(1).strip())
    if s.isdigit():
        return s
    # Bare public id (no query string)
    if "?" not in s and ":" not in s:
        return s
    return s


def event_ids_from_catalog(catalog: Catalog, *, dedupe: bool = True) -> list[str]:
    """Collect resource id strings from events; optional order-preserving dedupe."""
    out: list[str] = []
    seen: set[str] = set()
    for ev in catalog:
        rid = getattr(getattr(ev, "resource_id", None), "id", None)
        if not rid:
            continue
        s = str(rid).strip()
        if not s:
            continue
        if dedupe:
            if s in seen:
                continue
            seen.add(s)
        out.append(s)
    return out


def event_ids_for_detail_fetch(
    catalog: Catalog,
    *,
    dedupe: bool = True,
    normalize: bool = True,
) -> list[str]:
    """
    Like ``event_ids_from_catalog`` but normalize ids for ``get_events(eventid=...)``
    and dedupe on the **normalized** id (so duplicate events that differ only by URI
    form collapse to one detail request).
    """
    out: list[str] = []
    seen: set[str] = set()
    for ev in catalog:
        rid = getattr(getattr(ev, "resource_id", None), "id", None)
        if not rid:
            continue
        raw = str(rid).strip()
        if not raw:
            continue
        key = fdsn_eventid_for_get_events(raw) if normalize else raw
        if not key:
            continue
        if dedupe:
            if key in seen:
                continue
            seen.add(key)
        out.append(key)
    return out


def fetch_events_detail_merged(
    client: Client,
    event_ids: list[str],
    *,
    delay_seconds: float = 0.25,
    includearrivals: bool = True,
    includeallorigins: bool = True,
    includeallmagnitudes: bool = True,
    supports_include_arrivals: bool = True,
    progress: bool = False,
) -> tuple[Catalog, list[tuple[str, str]]]:
    """
    For each ``event_id``, call ``get_events(eventid=..., ...)`` and merge catalogs.

    If ``progress`` is True, print one line per request (eventid, response size,
    pick count, arrival count) to stdout.

    Returns
    -------
    merged : Catalog
    failures : list of (event_id, error_message)
    """
    merged = Catalog()
    failures: list[tuple[str, str]] = []
    total = len(event_ids)
    for i, eid in enumerate(event_ids):
        if delay_seconds > 0 and i > 0:
            time.sleep(delay_seconds)
        try:
            req: dict[str, Any] = {
                "eventid": eid,
                "includeallorigins": includeallorigins,
                "includeallmagnitudes": includeallmagnitudes,
            }
            if supports_include_arrivals:
                req["includearrivals"] = includearrivals
            cat = client.get_events(**req)
            if cat is None or len(cat) == 0:
                failures.append((eid, "empty response"))
                if progress:
                    print(f"[{i + 1}/{total}] eventid={eid} EMPTY", flush=True)
                continue
            np, na = _count_picks_and_arrivals(cat)
            if progress:
                print(
                    f"[{i + 1}/{total}] eventid={eid} events={len(cat)} picks={np} origin_arrivals={na}",
                    flush=True,
                )
            merged += cat
        except Exception as e:
            failures.append((eid, str(e)))
            _log.warning("Detail fetch failed for %s: %s", eid, e)
            if progress:
                print(f"[{i + 1}/{total}] eventid={eid} ERROR {e!s}", flush=True)
    return merged, failures


def download_fdsn_catalog_bulk_then_detail(
    output_quakeml: str,
    datasource: str = "IRIS",
    fdsn_endpoint: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    *,
    delay_seconds: float = 0.25,
    bulk_include_arrivals: bool = False,
    progress: bool = False,
    comcat_phases: bool = False,
    comcat_search_catalog: Optional[str] = None,
    comcat_phase_source: str = "preferred",
    comcat_delay_seconds: float = 0.25,
    comcat_progress: bool = False,
    **event_query_bulk: Any,
) -> tuple[Path, Catalog, list[tuple[str, str]], list[tuple[str, str]]]:
    """
    1) Bulk ``get_events`` (same filters as normal; do not pass ``eventid``).
    2) Extract deduplicated event ids.
    3) Per-event detail fetch with arrivals, rate-limited, merged.

    Parameters
    ----------
    event_query_bulk
        Passed to bulk ``get_events`` (e.g. starttime, endtime, latitude, maxradius).
        ``eventid`` is ignored for the bulk step if present.
    bulk_include_arrivals
        If False (default), bulk request omits arrivals for a lighter first response.
    delay_seconds
        Sleep between per-event requests (0 to disable).
    """
    resolved = _resolve_datasource(datasource, fdsn_endpoint)
    client_kwargs: dict[str, Any] = {}
    if user:
        client_kwargs["user"] = user
    if password:
        client_kwargs["password"] = password
    client = Client(resolved, **client_kwargs)

    supports_ia = datasource_supports_include_arrivals(datasource, fdsn_endpoint)

    bulk_kw = dict(event_query_bulk)
    bulk_kw.pop("eventid", None)
    if supports_ia:
        bulk_kw["includearrivals"] = bulk_include_arrivals
    else:
        bulk_kw.pop("includearrivals", None)
    bulk_kw.setdefault("includeallorigins", event_query_bulk.get("includeallorigins", True))
    bulk_kw.setdefault("includeallmagnitudes", event_query_bulk.get("includeallmagnitudes", True))

    _log.info("FDSN bulk list: %s", bulk_kw)
    bulk_cat = client.get_events(**bulk_kw)
    ids = event_ids_for_detail_fetch(bulk_cat, dedupe=True, normalize=True)
    _log.info("Bulk returned %d events, %d unique detail eventids", len(bulk_cat), len(ids))

    merged, failures = fetch_events_detail_merged(
        client,
        ids,
        delay_seconds=delay_seconds,
        includearrivals=True,
        includeallorigins=True,
        includeallmagnitudes=True,
        supports_include_arrivals=supports_ia,
        progress=progress,
    )

    comcat_failures: list[tuple[str, str]] = []
    if comcat_phases:
        from wav2hyp.tools.comcat_phases import enrich_catalog_with_comcat_phases

        merged, comcat_failures = enrich_catalog_with_comcat_phases(
            merged,
            comcat_search_catalog=comcat_search_catalog,
            comcat_phase_source=comcat_phase_source,
            delay_seconds=comcat_delay_seconds,
            progress=comcat_progress,
        )

    out = Path(output_quakeml)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.write(str(out), format="QUAKEML")
    return out, merged, failures, comcat_failures
