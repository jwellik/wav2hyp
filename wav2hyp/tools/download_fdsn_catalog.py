"""
Download an event catalog (with arrivals) from an FDSN service and write QuakeML.

Optionally enrich each event with phase picks from USGS ComCat (``usgs-libcomcat``),
using the ``phase-data`` product QuakeML (see :mod:`wav2hyp.tools.comcat_phases`).

Usage example:
  python -m wav2hyp.tools.download_fdsn_catalog \
    --starttime 2023-01-01 --endtime 2023-01-02 \
    --datasource IRIS \
    --output-quakeml catalog.xml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from obspy.clients.fdsn import Client

_log = logging.getLogger("wav2hyp")


def _default_comcat_phases(datasource: str, fdsn_endpoint: Optional[str]) -> bool:
    """Prefer ComCat phases for USGS FDSN; IRIS/global catalogs use FDSN picks only unless opted in."""
    ep = (fdsn_endpoint or "").lower()
    if "usgs" in ep or "earthquake.usgs.gov" in ep:
        return True
    return (datasource or "").strip().upper() == "USGS"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download FDSN event catalog with arrivals to QuakeML.")
    p.add_argument("--starttime", required=False, help="Start time (UTCDateTime compatible).")
    p.add_argument("--endtime", required=False, help="End time (UTCDateTime compatible).")
    p.add_argument("--eventid", required=False, help="Fetch a single event by FDSN event id.")

    p.add_argument("--minlatitude", type=float, default=None)
    p.add_argument("--maxlatitude", type=float, default=None)
    p.add_argument("--minlongitude", type=float, default=None)
    p.add_argument("--maxlongitude", type=float, default=None)
    p.add_argument("--latitude", type=float, default=None, help="Center latitude for radial query.")
    p.add_argument("--longitude", type=float, default=None, help="Center longitude for radial query.")
    p.add_argument("--minradius", type=float, default=None, help="Min radius in degrees.")
    p.add_argument("--maxradius", type=float, default=None, help="Max radius in degrees.")

    p.add_argument("--minmagnitude", type=float, default=None)
    p.add_argument("--maxmagnitude", type=float, default=None)
    p.add_argument("--mindepth", type=float, default=None)
    p.add_argument("--maxdepth", type=float, default=None)

    p.add_argument(
        "--datasource",
        default="IRIS",
        help="Named FDSN provider (e.g. IRIS, EARTHSCOPE, USGS, GFZ).",
    )
    p.add_argument(
        "--fdsn-endpoint",
        default=None,
        help="Custom FDSN endpoint (host:port or full URL). Overrides --datasource when set.",
    )
    p.add_argument("--user", default=None, help="Optional username for authenticated FDSN services.")
    p.add_argument("--password", default=None, help="Optional password for authenticated FDSN services.")

    p.add_argument("--output-quakeml", required=True, help="Output QuakeML file path.")
    p.add_argument(
        "--bulk-then-detail",
        action="store_true",
        help=(
            "Bulk list events first, then fetch each by eventid with includearrivals=True "
            "(better arrivals for many FDSN services). Uses --detail-delay-seconds between requests."
        ),
    )
    p.add_argument(
        "--detail-delay-seconds",
        type=float,
        default=0.25,
        help="Seconds to sleep between per-event detail requests (default: 0.25). Use 0 to disable.",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help=(
            "Per-event progress: FDSN detail (picks / origin arrivals) when using "
            "--bulk-then-detail; ComCat phase fetch when --comcat-phases is enabled."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")

    comcat = p.add_mutually_exclusive_group()
    comcat.add_argument(
        "--comcat-phases",
        dest="comcat_phases",
        action="store_const",
        const=True,
        default=None,
        help=(
            "After building the catalog, replace picks per event using USGS ComCat "
            "phase-data (requires ComCat id in resource_id, e.g. uw…, us…). "
            "Default: on for --datasource USGS or USGS endpoints."
        ),
    )
    comcat.add_argument(
        "--no-comcat-phases",
        dest="comcat_phases",
        action="store_const",
        const=False,
        default=None,
        help="Do not fetch ComCat phase-data (use only FDSN).",
    )
    p.add_argument(
        "--comcat-search-catalog",
        default=None,
        metavar="CATALOG",
        help="Optional ComCat catalog filter for get_event_by_id (e.g. atlas).",
    )
    p.add_argument(
        "--comcat-phase-source",
        default="preferred",
        help="phase-data product source: preferred, us, uw, ak, … (default: preferred).",
    )
    p.add_argument(
        "--comcat-delay-seconds",
        type=float,
        default=0.25,
        help="Delay between ComCat requests (default: 0.25). Use 0 to disable.",
    )
    return p.parse_args(argv)


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return f"http://{endpoint}"


def _resolve_datasource(datasource: str, fdsn_endpoint: Optional[str]) -> str:
    if fdsn_endpoint:
        return _normalize_endpoint(fdsn_endpoint)
    ds = (datasource or "").strip()
    # ObsPy FDSN shortcut compatibility: older environments know IRIS, not EARTHSCOPE.
    if ds.upper() == "EARTHSCOPE":
        return "IRIS"
    return ds


def _build_get_events_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "includearrivals": True,  # Required for pick/phase comparison workflows.
        "includeallorigins": True,
        "includeallmagnitudes": True,
    }
    for key in (
        "starttime",
        "endtime",
        "eventid",
        "minlatitude",
        "maxlatitude",
        "minlongitude",
        "maxlongitude",
        "latitude",
        "longitude",
        "minradius",
        "maxradius",
        "minmagnitude",
        "maxmagnitude",
        "mindepth",
        "maxdepth",
    ):
        value = getattr(args, key, None)
        if value is not None:
            kwargs[key] = value
    return kwargs


def _catalog_counts(catalog) -> tuple[int, int, int]:
    event_count = len(catalog)
    pick_count = 0
    arrival_count = 0
    for event in catalog:
        pick_count += len(getattr(event, "picks", None) or [])
        origin = event.preferred_origin() if hasattr(event, "preferred_origin") else None
        if origin is None:
            origins = getattr(event, "origins", None) or []
            origin = origins[0] if origins else None
        if origin is not None:
            arrival_count += len(getattr(origin, "arrivals", None) or [])
    return event_count, pick_count, arrival_count


def download_fdsn_catalog(
    output_quakeml: str,
    datasource: str = "IRIS",
    fdsn_endpoint: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    *,
    comcat_phases: bool = False,
    comcat_search_catalog: Optional[str] = None,
    comcat_phase_source: str = "preferred",
    comcat_delay_seconds: float = 0.25,
    comcat_progress: bool = False,
    **event_query,
) -> tuple[Path, object, list[tuple[str, str]]]:
    resolved_source = _resolve_datasource(datasource=datasource, fdsn_endpoint=fdsn_endpoint)
    client_kwargs = {}
    if user:
        client_kwargs["user"] = user
    if password:
        client_kwargs["password"] = password
    client = Client(resolved_source, **client_kwargs)
    eq = dict(event_query)
    from wav2hyp.tools.fdsn_detail_catalog import datasource_supports_include_arrivals

    if not datasource_supports_include_arrivals(datasource, fdsn_endpoint):
        eq.pop("includearrivals", None)
    catalog = client.get_events(**eq)

    comcat_failures: list[tuple[str, str]] = []
    if comcat_phases:
        from wav2hyp.tools.comcat_phases import enrich_catalog_with_comcat_phases

        catalog, comcat_failures = enrich_catalog_with_comcat_phases(
            catalog,
            comcat_search_catalog=comcat_search_catalog,
            comcat_phase_source=comcat_phase_source,
            delay_seconds=comcat_delay_seconds,
            progress=comcat_progress,
        )

    out = Path(output_quakeml)
    out.parent.mkdir(parents=True, exist_ok=True)
    catalog.write(str(out), format="QUAKEML")
    return out, catalog, comcat_failures


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger("wav2hyp").setLevel(logging.DEBUG)

    if args.comcat_phases is None:
        args.comcat_phases = _default_comcat_phases(args.datasource, args.fdsn_endpoint)

    if not args.eventid and (args.starttime is None or args.endtime is None):
        print("Error: provide --eventid OR both --starttime and --endtime.", file=sys.stderr)
        return 1

    event_kwargs = _build_get_events_kwargs(args)
    from wav2hyp.tools.fdsn_detail_catalog import datasource_supports_include_arrivals

    if not datasource_supports_include_arrivals(args.datasource, args.fdsn_endpoint):
        event_kwargs.pop("includearrivals", None)

    resolved_source = _resolve_datasource(args.datasource, args.fdsn_endpoint)
    try:
        if args.bulk_then_detail:
            from wav2hyp.tools.fdsn_detail_catalog import download_fdsn_catalog_bulk_then_detail

            # Bulk step uses lighter includearrivals; per-event step forces arrivals.
            event_kwargs.pop("includearrivals", None)
            out, catalog, failures, comcat_failures = download_fdsn_catalog_bulk_then_detail(
                output_quakeml=args.output_quakeml,
                datasource=args.datasource,
                fdsn_endpoint=args.fdsn_endpoint,
                user=args.user,
                password=args.password,
                delay_seconds=float(args.detail_delay_seconds),
                bulk_include_arrivals=False,
                progress=bool(args.progress),
                comcat_phases=bool(args.comcat_phases),
                comcat_search_catalog=args.comcat_search_catalog,
                comcat_phase_source=args.comcat_phase_source,
                comcat_delay_seconds=float(args.comcat_delay_seconds),
                comcat_progress=bool(args.progress and args.comcat_phases),
                **event_kwargs,
            )
            if failures:
                print(f"WARNING: {len(failures)} event detail fetches failed.", file=sys.stderr)
                if args.verbose:
                    for eid, msg in failures[:20]:
                        print(f"  {eid}: {msg}", file=sys.stderr)
                    if len(failures) > 20:
                        print(f"  ... and {len(failures) - 20} more", file=sys.stderr)
            if comcat_failures:
                print(
                    f"WARNING: {len(comcat_failures)} ComCat phase-data fetches failed or skipped.",
                    file=sys.stderr,
                )
                if args.verbose:
                    for eid, msg in comcat_failures[:20]:
                        print(f"  comcat {eid}: {msg}", file=sys.stderr)
                    if len(comcat_failures) > 20:
                        print(f"  ... and {len(comcat_failures) - 20} more", file=sys.stderr)
        else:
            out, catalog, comcat_failures = download_fdsn_catalog(
                output_quakeml=args.output_quakeml,
                datasource=args.datasource,
                fdsn_endpoint=args.fdsn_endpoint,
                user=args.user,
                password=args.password,
                comcat_phases=bool(args.comcat_phases),
                comcat_search_catalog=args.comcat_search_catalog,
                comcat_phase_source=args.comcat_phase_source,
                comcat_delay_seconds=float(args.comcat_delay_seconds),
                comcat_progress=bool(args.progress and args.comcat_phases),
                **event_kwargs,
            )
            if comcat_failures:
                print(
                    f"WARNING: {len(comcat_failures)} ComCat phase-data fetches failed or skipped.",
                    file=sys.stderr,
                )
                if args.verbose:
                    for eid, msg in comcat_failures[:20]:
                        print(f"  comcat {eid}: {msg}", file=sys.stderr)
                    if len(comcat_failures) > 20:
                        print(f"  ... and {len(comcat_failures) - 20} more", file=sys.stderr)
    except Exception as e:
        print(f"Error: failed to download catalog from {resolved_source}: {e}", file=sys.stderr)
        return 1

    print(f"Wrote QuakeML to {out}")
    print(f"Datasource: {resolved_source}")
    if args.fdsn_endpoint:
        print("Datasource selection: using --fdsn-endpoint override over --datasource")

    try:
        n_events, n_picks, n_arrivals = _catalog_counts(catalog)
        print(f"Counts: events={n_events}, picks={n_picks}, arrivals={n_arrivals}")
        if n_arrivals == 0:
            print("WARNING: Catalog has zero arrivals.", file=sys.stderr)
    except Exception:
        # Non-fatal; output already written.
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

