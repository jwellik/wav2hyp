"""
Catalog comparer: partition matched vs unmatched events, shared-arrivals tables,
and Matplotlib / VolcanoFigure visualizations (Canonical vs Test).

See ``docs/plan-catalog-comparison-plots.md``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from obspy import read_inventory
from obspy.geodetics import gps2dist_azimuth
from obspy import UTCDateTime
from obspy.core.event import Magnitude, Origin, Event, Catalog
from obspy.core.event import ResourceIdentifier

from vdapseisutils import VolcanoFigure

from wav2hyp.utils.io import NLLOutput
from wav2hyp.viz.plot_styles import (
    ARRIVAL_DELTA_T_XLIM,
    COLORS_80S,
    DELTA_DEPTH_CMAP_80S,
    META_BOTH_COLOR,
    META_CANONICAL_ONLY_COLOR,
    META_TEST_ONLY_COLOR,
    P_PHASE_80S,
    S_PHASE_80S,
    NLL_CATALOG_COLOR,
    apply_mpl_axes_style,
)

# Saved alongside notebooks/<volcano>/
FIG_NAME_TIMEBIN = "catalog_comparer__timebin_event_rate.png"
FIG_NAME_VOLCANO = "catalog_comparer__volcano_overlay.png"
FIG_NAME_POLAR = "catalog_comparer__polar_origin_offset.png"
FIG_NAME_ARRIVAL_SCATTER = "catalog_comparer__arrival_scatter_p_s.png"
FIG_NAME_ARRIVAL_BOXPLOT = "catalog_comparer__arrival_delta_by_station.png"


def default_notebook_export_dir(config_path: Path) -> Path:
    """``notebooks/<config_stem>/`` relative to repository root (like other explorers)."""
    stem = Path(config_path).resolve().stem
    anchor = Path(config_path).resolve().parent
    for base in (anchor, anchor.parent, *anchor.parents):
        n = base / "notebooks" / stem
        if n.parent.is_dir() or (base / "notebooks").is_dir():
            return n
    return anchor / "notebooks" / stem


def load_locator_tables(
    locator_h5: Path, *, t1: str | None, t2: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``catalog_table`` and time-bounded ``arrivals_table`` from a locator H5 file."""
    nll = NLLOutput(str(locator_h5))
    cat_df = nll.read_catalog_table(t1=t1, t2=t2).copy()
    if not cat_df.empty:
        cat_df["event_id"] = cat_df["event_id"].astype(str)
        cat_df["origin_time"] = pd.to_datetime(cat_df["origin_time"], utc=True, errors="coerce")
        cat_df = cat_df.sort_values("origin_time").reset_index(drop=True)
    arr_df = nll.read_arrivals(event_ids=[], t1=t1, t2=t2).copy()
    if not arr_df.empty:
        arr_df["event_id"] = arr_df["event_id"].astype(str)
        arr_df["arrival_time"] = pd.to_datetime(arr_df["arrival_time"], utc=True, errors="coerce")
        arr_df = arr_df.sort_values("arrival_time").reset_index(drop=True)
    return cat_df, arr_df


def apply_vdapseis_timeseries_axis_style(ax, *, grid: bool = False) -> None:
    """
    Match :class:`vdapseisutils.core.maps.time_series.TimeSeries` tick, spine, and
    label styling. Call :func:`obspy.imaging.util._set_xaxis_obspy_dates` on the
    same axes after plotting time-based data.
    """
    try:
        from vdapseisutils.core.maps.defaults import ensure_maps_mpl_style
        from vdapseisutils.core.maps.defaults import TICK_DEFAULTS, AXES_DEFAULTS
    except ImportError:
        ax.set_facecolor("white")
        ax.grid(grid)
        return
    ensure_maps_mpl_style()
    ax.set_facecolor("white")
    ax.grid(grid)
    for s in ax.spines.values():
        s.set_linewidth(AXES_DEFAULTS.get("spine_linewidth", 1.5))
    ax.tick_params(
        axis="both",
        labelcolor=TICK_DEFAULTS["labelcolor"],
        labelsize=TICK_DEFAULTS["labelsize"],
        color=TICK_DEFAULTS["tick_color"],
        length=TICK_DEFAULTS["tick_size"],
        width=TICK_DEFAULTS["tick_width"],
        direction=TICK_DEFAULTS["tick_direction"],
        pad=TICK_DEFAULTS["tick_pad"],
    )


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    filename: str,
    *,
    volcano_slug: str,
    dpi: int = 200,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{Path(filename).stem}__{volcano_slug}{Path(filename).suffix}"
    path = out_dir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _cat_df_to_obspy_catalog_local(df: pd.DataFrame) -> Catalog:
    """Build minimal ObsPy :class:`Catalog` from a wav2hyp ``catalog_table`` dataframe."""
    obspy_cat = Catalog()
    for _, r in df.iterrows():
        ot = pd.to_datetime(r["origin_time"], utc=True, errors="coerce")
        if pd.isna(ot):
            continue
        ev = Event(
            resource_id=ResourceIdentifier(id=str(r.get("event_id", "unknown"))),
        )
        ev_origin = Origin(
            resource_id=ResourceIdentifier(),
            time=UTCDateTime(ot.to_pydatetime()),
            latitude=float(r["latitude"]),
            longitude=float(r["longitude"]),
            depth=float(r["depth_km"]) * 1000.0,
        )
        mag_val = pd.to_numeric(r.get("mag"), errors="coerce")
        if pd.isna(mag_val):
            mag_val = 1.0
        mag = Magnitude(mag=float(mag_val), magnitude_type="Md", resource_id=ResourceIdentifier())
        ev.origins = [ev_origin]
        ev.magnitudes = [mag]
        obspy_cat.append(ev)
    return obspy_cat


def match_events_one_to_one(
    catalog_canonical: pd.DataFrame,
    catalog_test: pd.DataFrame,
    *,
    tolerance_s: float,
) -> pd.DataFrame:
    """
    Nearest-in-time match from canonical → test, then de-duplicate to reduce one-to-many pairings.
    """
    if catalog_canonical.empty or catalog_test.empty:
        return pd.DataFrame(
            columns=[
                "event_id_canonical",
                "event_id_test",
                "origin_time_canonical",
                "origin_time_test",
                "dt_s",
            ]
        )

    left = catalog_canonical[["event_id", "origin_time"]].rename(
        columns={"event_id": "event_id_canonical", "origin_time": "origin_time_canonical"}
    )
    right = catalog_test[["event_id", "origin_time"]].rename(
        columns={"event_id": "event_id_test", "origin_time": "origin_time_test"}
    )
    m = pd.merge_asof(
        left.sort_values("origin_time_canonical"),
        right.sort_values("origin_time_test"),
        left_on="origin_time_canonical",
        right_on="origin_time_test",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_s),
    )
    m = m.dropna(subset=["event_id_test", "event_id_canonical"])
    m["dt_s"] = (m["origin_time_canonical"] - m["origin_time_test"]).dt.total_seconds()
    m["absdt"] = m["dt_s"].abs()
    m = m.sort_values("absdt")
    m = m.drop_duplicates("event_id_test", keep="first")
    m = m.sort_values("absdt")
    m = m.drop_duplicates("event_id_canonical", keep="first")
    return m.reset_index(drop=True)


def partition_by_match(
    catalog_canonical: pd.DataFrame,
    catalog_test: pd.DataFrame,
    matched_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return ``(canonical_only, test_only, both_at_canonical)``.

    ``both_at_canonical`` is one row per match at **canonical** hypocenter (for a single map layer).
    """
    ec = set(matched_pairs["event_id_canonical"].astype(str)) if len(matched_pairs) else set()
    et = set(matched_pairs["event_id_test"].astype(str)) if len(matched_pairs) else set()

    c = catalog_canonical.copy()
    t = catalog_test.copy()
    c["event_id"] = c["event_id"].astype(str)
    t["event_id"] = t["event_id"].astype(str)

    canonical_only = c[~c["event_id"].isin(ec)].copy()
    test_only = t[~t["event_id"].isin(et)].copy()
    if not ec:
        return canonical_only, test_only, c.iloc[0:0].copy()
    both_at_canonical = c[c["event_id"].isin(ec)].copy()
    return canonical_only, test_only, both_at_canonical


def enrich_matched_pairs_locations(
    matched_pairs: pd.DataFrame,
    catalog_canonical: pd.DataFrame,
    catalog_test: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``latitude_*``, ``longitude_*``, ``depth_km_*`` columns for polar / summary plots."""
    c = catalog_canonical.copy()
    t = catalog_test.copy()
    c["event_id"] = c["event_id"].astype(str)
    t["event_id"] = t["event_id"].astype(str)
    m = matched_pairs.copy()
    m = m.merge(
        c[["event_id", "latitude", "longitude", "depth_km"]].rename(
            columns={
                "event_id": "event_id_canonical",
                "latitude": "latitude_canonical",
                "longitude": "longitude_canonical",
                "depth_km": "depth_km_canonical",
            }
        ),
        on="event_id_canonical",
        how="left",
    )
    m = m.merge(
        t[["event_id", "latitude", "longitude", "depth_km"]].rename(
            columns={
                "event_id": "event_id_test",
                "latitude": "latitude_test",
                "longitude": "longitude_test",
                "depth_km": "depth_km_test",
            }
        ),
        on="event_id_test",
        how="left",
    )
    return m


def _norm_station_id(station_id: object) -> str:
    s = str(station_id) if station_id is not None else ""
    s = s.strip()
    if not s:
        return ""
    parts = s.split(".")
    if len(parts) >= 2 and len(parts[1]) >= 2:
        return parts[1]  # NET.STA.LOC.CHA or STA.LOC
    return parts[0] if parts else s


def phase_hint_from_phase(phase: object) -> str | None:
    ph = (str(phase).strip().upper() if phase is not None else "")
    if not ph:
        return None
    if ph.startswith("P"):
        return "P"
    if ph.startswith("S"):
        return "S"
    return None


def _dedupe_arrivals(
    arr: pd.DataFrame,
    *,
    event_id_col: str,
    sub_keys: list[str],
) -> pd.DataFrame:
    a = arr.copy()
    a = a.sort_values([event_id_col, *sub_keys, "arrival_time"], na_position="last")
    return a.drop_duplicates([event_id_col, *sub_keys], keep="first").copy()


def build_shared_arrivals(
    matched_pairs: pd.DataFrame,
    arrivals_canonical: pd.DataFrame,
    arrivals_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join on ``(event_id pair, station_key, phase_hint)``; one row per key after per-side de-dupe.
    """
    if len(matched_pairs) == 0 or arrivals_canonical.empty or arrivals_test.empty:
        return pd.DataFrame(
            columns=[
                "event_id_canonical",
                "event_id_test",
                "station_id_canonical",
                "station_id_test",
                "station_key",
                "phase_hint",
                "pick_id_canonical",
                "pick_id_test",
                "arrival_time_canonical",
                "arrival_time_test",
                "arrival_time_delta",
            ]
        )

    pairs = matched_pairs[
        ["event_id_canonical", "event_id_test"]
    ].drop_duplicates()
    ac = arrivals_canonical.copy()
    at = arrivals_test.copy()
    ac["event_id"] = ac["event_id"].astype(str)
    at["event_id"] = at["event_id"].astype(str)
    for df in (ac, at):
        if "station_id" not in df.columns:
            raise KeyError("arrivals need station_id")
        if "arrival_time" not in df.columns or "pick_id" not in df.columns:
            raise KeyError("arrivals need arrival_time and pick_id")
        if "phase" not in df.columns:
            raise KeyError("arrivals need phase")

    ac = ac.merge(
        pairs,
        left_on="event_id",
        right_on="event_id_canonical",
        how="inner",
    )
    at = at.merge(
        pairs,
        left_on="event_id",
        right_on="event_id_test",
        how="inner",
    )
    for df in (ac, at):
        df["station_key"] = df["station_id"].map(_norm_station_id)
        df["phase_hint"] = df["phase"].map(phase_hint_from_phase)
    ac = ac[ac["phase_hint"].notna()].copy()
    at = at[at["phase_hint"].notna()].copy()

    sub = ["event_id_canonical", "event_id_test", "station_key", "phase_hint"]
    ac2 = _dedupe_arrivals(
        ac,
        event_id_col="event_id_canonical",
        sub_keys=["event_id_test", "station_key", "phase_hint"],
    )
    at2 = _dedupe_arrivals(
        at,
        event_id_col="event_id_test",
        sub_keys=["event_id_canonical", "station_key", "phase_hint"],
    )

    # ``event_id`` exists on both sides; merge suffixes would create ``event_id_canonical`` /
    # ``event_id_test`` and clash with the real pair-id columns.
    ac2 = ac2.drop(columns=["event_id"], errors="ignore")
    at2 = at2.drop(columns=["event_id"], errors="ignore")

    merged = ac2.merge(
        at2,
        on=["event_id_canonical", "event_id_test", "station_key", "phase_hint"],
        how="inner",
        suffixes=("_canonical", "_test"),
    )
    for col in ("arrival_time_canonical", "arrival_time_test"):
        if col not in merged.columns:
            raise KeyError("merge should produce suffixed arrival_time columns")
    merged["arrival_time_canonical"] = pd.to_datetime(
        merged["arrival_time_canonical"], utc=True, errors="coerce"
    )
    merged["arrival_time_test"] = pd.to_datetime(
        merged["arrival_time_test"], utc=True, errors="coerce"
    )
    merged["arrival_time_delta"] = (
        merged["arrival_time_test"] - merged["arrival_time_canonical"]
    ).dt.total_seconds()
    for c in ("pick_id_canonical", "pick_id_test", "station_id_canonical", "station_id_test"):
        if c not in merged.columns:
            raise KeyError(
                f"build_shared_arrivals: expected column {c} from merge; got {list(merged.columns)[:20]}…"
            )
    merged["arrival_id_canonical"] = merged["pick_id_canonical"].astype(str)
    merged["arrival_id_test"] = merged["pick_id_test"].astype(str)
    return merged.reset_index(drop=True)


def enrich_shared_arrivals_with_eqt_peak_value(
    shared: pd.DataFrame,
    *,
    eqt_volpick_h5: Path | None,
    t1: str | None,
    t2: str | None,
) -> pd.DataFrame:
    """
    Add ``peak_value_canonical`` from ``eqt-volpick.h5`` picks (nearest
    ``peak_time`` to canonical ``arrival_time``, same station + P/S family) via
    :func:`wav2hyp.viz.sthelens_clipboards.attach_pick_probabilities`.
    """
    out = shared.copy()
    if not eqt_volpick_h5 or not Path(eqt_volpick_h5).is_file():
        return out
    need = ("station_id_canonical", "phase", "arrival_time_canonical")
    if not all(c in out.columns for c in need):
        return out
    mini = out[list(need)].copy().rename(
        columns={"station_id_canonical": "station_id", "arrival_time_canonical": "arrival_time"}
    )
    from wav2hyp.viz.sthelens_clipboards import attach_pick_probabilities

    tagged = attach_pick_probabilities(
        pd.DataFrame(),
        mini,
        eqt_volpick_h5=Path(eqt_volpick_h5),
        t1=t1,
        t2=t2,
    )
    out["peak_value_canonical"] = pd.to_numeric(tagged["prob"], errors="coerce")
    return out


def attach_y_metric_for_plots(
    shared: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    """
    Y column for scatter: prefer *peak_value* (canonical, then test), else *weight*, else *residual*.
    Returns (df with ``y_plot``, y-axis label, short name for titles).
    """
    out = shared.copy()
    s: pd.Series = pd.Series(np.nan, index=out.index)
    y_label = "metric"
    y_name = "metric"
    for col, nm, lab in (
        ("peak_value_canonical", "peak_value", "wav2hyp (peak_value)"),
        ("peak_value_test", "peak_value", "wav2hyp (peak_value)"),
        ("weight_canonical", "weight", "NLL time_weight"),
        ("weight_test", "weight", "NLL time_weight"),
        ("residual_canonical", "residual", "NLL residual (s)"),
        ("residual_test", "residual", "NLL residual (s)"),
    ):
        if col in out.columns and pd.to_numeric(out[col], errors="coerce").notna().any():
            s = pd.to_numeric(out[col], errors="coerce")
            y_label = lab
            y_name = nm
            break
    out["y_plot"] = s
    return out, y_label, y_name


def plot_timebin_stairs(
    catalog_canonical: pd.DataFrame,
    catalog_test: pd.DataFrame,
    *,
    freq: str,
    label_canonical: str = "Canonical",
    label_test: str = "Test",
) -> plt.Figure:
    c = catalog_canonical.copy()
    t = catalog_test.copy()
    for d in (c, t):
        d["origin_time"] = pd.to_datetime(d["origin_time"], utc=True, errors="coerce")
    c = c.dropna(subset=["origin_time"])
    t = t.dropna(subset=["origin_time"])
    if c.empty and t.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        apply_vdapseis_timeseries_axis_style(ax, grid=False)
        ax.text(0.5, 0.5, "no events", ha="center", va="center", transform=ax.transAxes)
        return fig

    # ``min(..., default=)`` is only valid for a single iterable, not two timestamps.
    oc = c["origin_time"]
    ot = t["origin_time"]
    if c.empty:
        tmin, tmax = ot.min(), ot.max()
    elif t.empty:
        tmin, tmax = oc.min(), oc.max()
    else:
        tmin = min(oc.min(), ot.min())
        tmax = max(oc.max(), ot.max())
    idx = pd.date_range(
        tmin.floor(freq),
        tmax.ceil(freq),
        freq=freq,
        inclusive="left",
    )

    def counts(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(0, index=idx[:-1], dtype="int64")
        s = (
            df.set_index("origin_time")
            .sort_index()
            .groupby(pd.Grouper(freq=freq))
            .size()
        )
        return s.reindex(idx[:-1], fill_value=0).astype("int64")

    sc, st = counts(c), counts(t)
    edges = idx

    fig, ax = plt.subplots(figsize=(10, 4))
    apply_vdapseis_timeseries_axis_style(ax, grid=False)
    ax.stairs(
        sc.to_numpy(),
        edges,
        color=NLL_CATALOG_COLOR,
        label=label_canonical,
        linewidth=1.4,
    )
    ax.stairs(
        st.to_numpy(),
        edges,
        color=META_TEST_ONLY_COLOR,
        label=label_test,
        linewidth=1.4,
        linestyle="--",
    )
    try:
        from obspy.imaging.util import _set_xaxis_obspy_dates
        from vdapseisutils.core.maps.defaults import TICK_DEFAULTS, TITLE_DEFAULTS
    except ImportError:
        TICK_DEFAULTS = {"axes_labelcolor": "grey", "axes_labelsize": "medium", "labelcolor": "grey"}
        TITLE_DEFAULTS = {"fontsize": "medium", "color": "k", "fontweight": "normal"}
        def _set_xaxis_obspy_dates(a):
            return None

    _set_xaxis_obspy_dates(ax)
    ax.set_xlabel(
        "Time",
        color=TICK_DEFAULTS["axes_labelcolor"],
        fontsize=TICK_DEFAULTS["axes_labelsize"],
    )
    ax.set_ylabel(
        f"Events per {freq}",
        color=TICK_DEFAULTS["axes_labelcolor"],
        fontsize=TICK_DEFAULTS["axes_labelsize"],
    )
    ax.set_title(
        "Event rate (time bins)",
        fontsize=TITLE_DEFAULTS.get("fontsize", "medium"),
        color=TITLE_DEFAULTS.get("color", "k"),
        fontweight=TITLE_DEFAULTS.get("fontweight", "normal"),
    )
    ax.set_ylim(bottom=0.0)
    leg = ax.legend(
        loc="upper right",
        frameon=True,
    )
    if leg is not None:
        for t in leg.get_texts():
            t.set_color(TICK_DEFAULTS.get("labelcolor", "grey"))
    try:
        fig.autofmt_xdate()
    except Exception:
        pass
    return fig


def plot_polar_origin_offset(
    matched_with_depth: pd.DataFrame,
    *,
    title: str | None = None,
    delta_depth_vmax_km: float | None = None,
) -> plt.Figure:
    """
    ``matched_with_depth`` must include
    ``latitude_canonical, longitude_canonical, latitude_test, longitude_test, depth_km_canonical, depth_km_test``.
    """
    df = matched_with_depth.copy()
    need = [
        "latitude_canonical",
        "longitude_canonical",
        "latitude_test",
        "longitude_test",
    ]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"plot_polar_origin_offset: missing {c}")
    ddc = "depth_km_canonical" if "depth_km_canonical" in df.columns else "depth_km"
    ddt = "depth_km_test" if "depth_km_test" in df.columns else "depth_km_t"
    if ddc not in df.columns or ddt not in df.columns:
        df["delta_depth_km"] = 0.0
    else:
        df["delta_depth_km"] = pd.to_numeric(df[ddt], errors="coerce") - pd.to_numeric(
            df[ddc], errors="coerce"
        )

    lat1 = np.asarray(df["latitude_canonical"], dtype="float64")
    lon1 = np.asarray(df["longitude_canonical"], dtype="float64")
    lat2 = np.asarray(df["latitude_test"], dtype="float64")
    lon2 = np.asarray(df["longitude_test"], dtype="float64")
    m = np.isfinite(lat1) & np.isfinite(lon1) & np.isfinite(lat2) & np.isfinite(lon2)
    r_km = np.full(len(df), np.nan, dtype="float64")
    az_deg = np.full(len(df), np.nan, dtype="float64")
    for i in np.where(m)[0]:
        d_m, az, _ = gps2dist_azimuth(
            float(lat1[i]), float(lon1[i]), float(lat2[i]), float(lon2[i])
        )
        r_km[i] = d_m / 1000.0
        az_deg[i] = float(az)

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    th = np.deg2rad(az_deg)
    c = np.asarray(df["delta_depth_km"], dtype="float64")
    c_abs = float(np.nanmax(np.abs(c[np.isfinite(c)]))) if np.isfinite(c).any() else 1.0
    vlim = (
        float(delta_depth_vmax_km)
        if delta_depth_vmax_km is not None
        else min(100.0, max(0.5, c_abs))
    )
    from matplotlib.colors import Normalize

    sc = ax.scatter(
        th,
        r_km,
        c=c,
        cmap=DELTA_DEPTH_CMAP_80S,
        norm=Normalize(vmin=-vlim, vmax=vlim),
        s=20,
        alpha=0.88,
        edgecolors="0.2",
        linewidths=0.2,
    )
    cbar = fig.colorbar(
        sc,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        fraction=0.035,
        shrink=0.75,
        aspect=28,
    )
    cbar.set_label("Δ depth (km)", rotation=90, labelpad=10, fontsize="small", color="grey")
    cbar.ax.tick_params(labelsize="small", colors="grey")
    n_ok = int(np.sum(np.isfinite(r_km)))
    ax.set_title(
        title
        or f"Azimuth vs distance (colored by Δ depth) (n={n_ok} events)"
    )
    rk = r_km[np.isfinite(r_km)]
    # Use the full distance range (not a high percentile), otherwise the radial
    # limit can sit below the largest offsets and the plot looks "capped" (e.g. 1 km).
    if rk.size:
        r_hi = float(np.nanmax(rk)) * 1.12
        if r_hi <= 0.0 or not np.isfinite(r_hi):
            r_hi = 1.0
    else:
        r_hi = 1.0
    ax.set_ylim(0.0, r_hi)
    # Radial tick labels: show distance in km on every grid line
    r_ticks = np.asarray(ax.get_yticks(), dtype=float)
    r_ticks = r_ticks[(r_ticks > 0) & (r_ticks <= ax.get_ylim()[1] + 1e-9)]
    if r_ticks.size:
        ax.set_rgrids(
            r_ticks,
            labels=[f"{x:g} km" for x in r_ticks],
            angle=22.5,
        )
    return fig


def make_catalog_comparer_volcano_figure(
    cfg: dict,
    inventory_path: Path,
    df_canonical_only: pd.DataFrame,
    df_test_only: pd.DataFrame,
    df_both: pd.DataFrame,
    *,
    map_t1: str | None = None,
    map_t2: str | None = None,
) -> VolcanoFigure:
    """Map + 2 cross-sections; **no** time axis; three catalog overlays."""
    fig = plt.figure(
        FigureClass=VolcanoFigure,
        figsize=(10, 10),
        origin=(float(cfg["target"]["latitude"]), float(cfg["target"]["longitude"])),
        radial_extent_km=15.0,
        depth_extent=(-15, 5),
        ts_axis_type=None,
    )
    try:
        fig.add_terrain()
    except Exception as e:
        print(f"terrain tiles failed: {e}")
    try:
        fig.add_hillshade(alpha=0.6)
    except Exception as e:
        print(f"hillshade failed: {e}")
    inv = read_inventory(str(inventory_path))
    fig.plot_inventory(inv, s=35, c="black", alpha=0.9, cross_section_s=12)
    n_c = int(len(df_canonical_only))
    n_t = int(len(df_test_only))
    n_b = int(len(df_both))
    layers: list[tuple[pd.DataFrame, str, int, str, int]] = [
        (df_canonical_only, "Canonical only", n_c, META_CANONICAL_ONLY_COLOR, 2),
        (df_test_only, "Test only", n_t, META_TEST_ONLY_COLOR, 3),
        (df_both, "Both (at canonical)", n_b, META_BOTH_COLOR, 4),
    ]
    leg_handles: list[Line2D] = []
    for df_ly, label, n_ev, color, z in layers:
        obspy_cat = _cat_df_to_obspy_catalog_local(df_ly)
        if len(obspy_cat) > 0:
            fig.plot_catalog(
                obspy_cat,
                s=48,
                c=color,
                alpha=0.5,
                edgecolors="0.15",
                linewidths=0.35,
                zorder=z,
            )
        leg_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markerfacecolor=color,
                markeredgecolor="0.15",
                markeredgewidth=0.35,
                markersize=7.5,
                alpha=0.5,
                label=f"{label} (n={n_ev})",
            )
        )
    win = f"{map_t1} to {map_t2}" if map_t1 is not None and map_t2 is not None else "full window"
    try:
        fig.map_obj.ax.set_title(
            f"{cfg['target']['name']} | comparer overlay | {win}\n"
            f"canonical only n={n_c} · test only n={n_t} · both n={n_b}",
            fontsize=10,
        )
    except Exception:
        pass
    if leg_handles:
        fig.map_obj.ax.legend(
            handles=leg_handles,
            loc="lower right",
            fontsize=8,
            framealpha=0.9,
        )
    return fig


def _station_color_map_80s(stations: list[str]) -> dict[str, tuple]:
    ncols = len(COLORS_80S)
    return {s: mcolors.to_rgba(COLORS_80S[i % ncols]) for i, s in enumerate(stations)}


def plot_arrival_scatter_p_s_panels(
    shared: pd.DataFrame,
) -> plt.Figure:
    """
    One row, two columns: P-waves, S-waves. One scatter **series per station** (``station_key``).
    """
    work, y_lab, y_name = attach_y_metric_for_plots(shared)
    for col in ("y_plot", "arrival_time_delta", "station_key", "phase_hint"):
        if col not in work.columns and col in shared.columns:
            work[col] = shared[col]
    if "station_key" not in work.columns and "station_id_canonical" in work.columns:
        work["station_key"] = work["station_id_canonical"].map(_norm_station_id)
    st_list = sorted(work["station_key"].dropna().astype(str).unique().tolist())
    cmap = _station_color_map_80s(st_list)
    try:
        from vdapseisutils.core.maps.defaults import TICK_DEFAULTS
    except ImportError:
        TICK_DEFAULTS = {"axes_labelcolor": "grey", "axes_labelsize": "small"}
    fig, (axp, axs) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for a in (axp, axs):
        a.set_facecolor("white")
        a.grid(False)
        apply_vdapseis_timeseries_axis_style(a, grid=False)
        a.set_xlim(ARRIVAL_DELTA_T_XLIM)
    y_axis_label = "wav2hyp (EQT peak_value)" if y_name == "peak_value" else y_lab
    for ph, ax, wave_name in (("P", axp, "P-waves"), ("S", axs, "S-waves")):
        sub = work[work["phase_hint"] == ph]
        for sta in st_list:
            s2 = sub[sub["station_key"] == sta]
            if s2.empty:
                continue
            color = cmap.get(sta, (0.2, 0.2, 0.2, 1.0))
            ax.scatter(
                s2["arrival_time_delta"],
                s2["y_plot"],
                s=12,
                alpha=0.78,
                c=[color] * len(s2),
                edgecolors="none",
                label=sta,
            )
        ax.axvline(0.0, color="0.35", linewidth=1.0, zorder=0)
        ax.set_xlabel(
            r"$\Delta t$ (seconds)",
            color=TICK_DEFAULTS["axes_labelcolor"],
            fontsize=TICK_DEFAULTS["axes_labelsize"],
        )
        if ph == "P":
            ax.set_ylabel(
                y_axis_label,
                color=TICK_DEFAULTS["axes_labelcolor"],
                fontsize=TICK_DEFAULTS["axes_labelsize"],
            )
        ptitle = "EQT peak_value" if y_name == "peak_value" else y_lab
        ax.set_title(
            f"Arrival time delta v {ptitle} ({wave_name})",
            fontsize=10,
            color=TICK_DEFAULTS.get("axes_labelcolor", "k"),
        )
    h, lab = axp.get_legend_handles_labels()
    n = len(lab) if lab else 0
    ncols = min(5, max(1, n)) if n else 1
    if h:
        fig.legend(
            h,
            lab,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=ncols,
            fontsize=7.5,
            frameon=True,
        )
    fig.suptitle(
        "Arrival time delta v EQT peak_value (P-waves | S-waves), 80s palette by station"
        if y_name == "peak_value"
        else f"Arrival time delta v {y_lab} (P-waves | S-waves)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    return fig


def plot_arrival_boxplot_by_station(
    shared: pd.DataFrame,
) -> plt.Figure:
    """Horizontal boxplots: y = station, x = ``arrival_time_delta``; P/S in 80s palette."""
    import matplotlib.patches as mpatches2

    w = shared.copy()
    if "station_key" not in w.columns and "station_id_canonical" in w.columns:
        w["station_key"] = w["station_id_canonical"].map(_norm_station_id)
    w = w.dropna(subset=["arrival_time_delta", "station_key", "phase_hint"])
    st_order = sorted(w["station_key"].dropna().astype(str).unique().tolist())
    n_sta = max(1, len(st_order))
    row_h = 0.4
    fig, ax = plt.subplots(figsize=(7.0, 0.55 * n_sta + 1.2))
    ax.set_facecolor("white")
    ax.grid(False)
    apply_vdapseis_timeseries_axis_style(ax, grid=False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    y_center = 0.0
    y_tick_pos: list[float] = []
    n_ann: list[tuple[float, str, int, str]] = []
    for sta in st_order:
        psub = w[(w["station_key"] == sta) & (w["phase_hint"] == "P")]["arrival_time_delta"]
        ssub = w[(w["station_key"] == sta) & (w["phase_hint"] == "S")]["arrival_time_delta"]
        np_ = int(len(psub))
        ns_ = int(len(ssub))
        ppos = y_center - row_h
        spos = y_center + row_h
        for dat, pos, col, n in ((psub, ppos, P_PHASE_80S, np_), (ssub, spos, S_PHASE_80S, ns_)):
            if n > 0:
                b = ax.boxplot(
                    [dat.to_numpy(dtype="float64")],
                    vert=False,
                    positions=[pos],
                    widths=0.28,
                    patch_artist=True,
                    manage_ticks=False,
                )
                for box in b["boxes"]:
                    box.set_facecolor(col)
                    box.set_alpha(0.45)
            n_ann.append((pos, col, n, f"N = {n}"))
        y_tick_pos.append(y_center)
        y_center += 1.0

    ax.set_yticks(y_tick_pos, labels=st_order, fontsize=8)
    try:
        from vdapseisutils.core.maps.defaults import TICK_DEFAULTS
    except ImportError:
        TICK_DEFAULTS = {"axes_labelcolor": "grey", "axes_labelsize": "small"}
    ax.set_xlabel(
        r"$\Delta t$ (seconds)",
        color=TICK_DEFAULTS["axes_labelcolor"],
        fontsize=TICK_DEFAULTS["axes_labelsize"],
    )
    ax.set_ylabel(
        "Station",
        color=TICK_DEFAULTS["axes_labelcolor"],
        fontsize=TICK_DEFAULTS["axes_labelsize"],
    )
    ax.set_title(
        "Distribution of pick time differences by station",
        fontsize=10,
        color=TICK_DEFAULTS.get("axes_labelcolor", "k"),
    )
    ax.set_xlim(ARRIVAL_DELTA_T_XLIM)
    lim = max(10.0, float(ARRIVAL_DELTA_T_XLIM[1]))
    for pos, col, _n, txt in n_ann:
        ax.text(
            lim * 0.99,
            pos,
            txt,
            fontsize=6.5,
            color=col,
            ha="right",
            va="center",
        )
    ax.axvline(0.0, color="k", linewidth=1.0, zorder=0)
    ax.legend(
        [
            mpatches2.Patch(facecolor=P_PHASE_80S, edgecolor="0.2", label="P"),
            mpatches2.Patch(facecolor=S_PHASE_80S, edgecolor="0.2", label="S"),
        ],
        ["P", "S"],
        loc="lower right",
        fontsize=8,
    )
    return fig