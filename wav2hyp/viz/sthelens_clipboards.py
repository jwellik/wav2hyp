"""
St. Helens–style figures: VolcanoFigure catalog overview, per-event Map + CrossSection,
waveform clipboards, and combined PNG stacks.

Used by :mod:`wav2hyp.notebooks.catalog_explorer` and the optional analysis script
under ``analysis_local/``. Paths are passed in via :class:`StHelensVizPaths` or
:func:`sthelens_paths_from_wav2hyp_config`.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import yaml
from obspy import Stream, UTCDateTime, read_inventory
from obspy.clients.filesystem.sds import Client as SDSClient

from wav2hyp.config_loader import config_path_anchor
from wav2hyp.viz.plot_styles import NLL_CATALOG_COLOR, SCATTER_CLOUD_TEAL
from wav2hyp.utils.io import NLLOutput

from obspy.core.event import Catalog, Event, Origin, Magnitude

from vdapseisutils import VolcanoFigure
from vdapseisutils.core.maps.cross_section import CrossSection
from vdapseisutils.core.maps.map import Map
from vdapseisutils.core.swarmmpl.clipboard import ClipboardClass, t2axiscoords

import nllpy


@dataclass(frozen=True)
class StHelensVizPaths:
    """Resolved filesystem layout for St. Helens–style visualization."""

    config_path: Path
    config_root: Path
    run_base_dir: Path
    inventory_path: Path
    nll_h5: Path
    eqt_volpick_h5: Path
    nll_loc_root: Path
    nll_run_prefix: str


def sthelens_paths_from_wav2hyp_config(
    cfg: dict, *, config_path: Path, run_base_dir: Path
) -> StHelensVizPaths:
    """
    Build paths from a WAV2HYP YAML dict (same keys as ``examples/sthelens.yaml``).

    ``run_base_dir`` is the pipeline output root (``output.base_dir``), containing
    ``locations/nll.h5`` and ``picks/eqt-volpick.h5``.

    Relative ``inventory.file`` and ``locator.nll_home`` entries are resolved with
    :func:`config_path_anchor` (repository root), not the YAML file's directory.
    """
    config_path = config_path.resolve()
    anchor = config_path_anchor(config_path)
    inv = Path(cfg["inventory"]["file"])
    inventory_path = inv.resolve() if inv.is_absolute() else (anchor / inv).resolve()
    out = cfg["output"]
    nll_home = Path(cfg["locator"]["nll_home"])
    nll_loc_root = nll_home.resolve() if nll_home.is_absolute() else (anchor / nll_home).resolve()
    rb = run_base_dir.resolve()
    return StHelensVizPaths(
        config_path=config_path,
        config_root=anchor,
        run_base_dir=rb,
        inventory_path=inventory_path,
        nll_h5=(rb / out.get("locator_dir", "locations") / "nll.h5").resolve(),
        eqt_volpick_h5=(rb / out.get("picker_dir", "picks") / "eqt-volpick.h5").resolve(),
        nll_loc_root=nll_loc_root,
        nll_run_prefix=str(cfg["locator"]["config_name"]),
    )


def sthelens_paths_with_optional_legacy_subdirs(
    cfg: dict, *, config_path: Path, run_base_dir: Path
) -> StHelensVizPaths:
    """
    Like :func:`sthelens_paths_from_wav2hyp_config`, but if ``3_locations/nll.h5`` exists
    under ``run_base_dir`` (older numbered stage layout), use that and ``1_picks/eqt-volpick.h5``.
    """
    std = sthelens_paths_from_wav2hyp_config(cfg, config_path=config_path, run_base_dir=run_base_dir)
    rb = std.run_base_dir
    nll_alt = rb / "3_locations" / "nll.h5"
    if nll_alt.is_file():
        eqt_alt = rb / "1_picks" / "eqt-volpick.h5"
        return StHelensVizPaths(
            config_path=std.config_path,
            config_root=std.config_root,
            run_base_dir=rb,
            inventory_path=std.inventory_path,
            nll_h5=nll_alt,
            eqt_volpick_h5=eqt_alt if eqt_alt.is_file() else std.eqt_volpick_h5,
            nll_loc_root=std.nll_loc_root,
            nll_run_prefix=std.nll_run_prefix,
        )
    return std


# Max |peak_time − NLL arrival_time| (s) to accept an eqt-volpick pick as the match
EQT_PICK_MATCH_MAX_DT_S = 2.0

WAVEFORM_PAD_BEFORE_S = 5.0
WAVEFORM_PAD_AFTER_S = 25.0
BANDPASS = (1.0, 15.0)

# Per-event map: fixed on volcano target (not event epicenter)
EVENT_MAP_RADIUS_KM = 20.0
EVENT_MAP_DEPTH_EXTENT = (-10.0, 5.0)

# Clipboard layout (P/S α legend is a separate figure; only caption + suptitle here)
CLIPBOARD_SUBPLOTS_TOP = 0.88
CLIPBOARD_SUBPLOTS_BOTTOM = 0.12
CLIPBOARD_SUPTITLE_Y = 0.97
CLIPBOARD_CAPTION_Y = 0.06


def load_config_from_path(config_path: Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_catalog_and_arrivals(
    nll_h5: Path | str, t1: str | None, t2: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = NLLOutput(str(nll_h5))
    cat = out.read_catalog_table(t1=t1, t2=t2).copy()
    cat["origin_time"] = pd.to_datetime(cat["origin_time"], utc=True)
    cat.sort_values("origin_time", inplace=True, ignore_index=True)
    eids = cat["event_id"].tolist()
    # NLLOutput.read_arrivals builds a PyTables OR over event_id; long chains hit numexpr limits.
    batch_size = 20
    parts: list[pd.DataFrame] = []
    for i in range(0, len(eids), batch_size):
        parts.append(out.read_arrivals(event_ids=eids[i : i + batch_size]))
    arr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return cat, arr


def _station_code_from_id(trace_id_or_station_id: str) -> str:
    # NLL station_id ".PPCL" or trace_id ".PPCL..Z" -> "PPCL"
    # pyocto station "VG.PPCL.00" or "VG.PPKB.1J" -> "PPCL"
    parts = str(trace_id_or_station_id).split(".")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return str(trace_id_or_station_id)


def attach_pick_probabilities(
    _cat_df: pd.DataFrame,
    arr_df: pd.DataFrame,
    *,
    eqt_volpick_h5: Path,
    t1: str | None,
    t2: str | None,
) -> pd.DataFrame:
    """Attach eqt-volpick pick probability (`peak_value`) to each NLL arrival row.

    For each arrival, finds the picker row with the same station and P/S family
    whose ``peak_time`` is closest to the NLL ``arrival_time`` (within
    ``EQT_PICK_MATCH_MAX_DT_S``). Returns a copy of arr_df with column ``prob``.
    """
    arr_out = arr_df.copy()
    arr_out["prob"] = np.nan

    try:
        picks_df = pd.read_hdf(str(eqt_volpick_h5), key="picks")
    except Exception as e:
        print(f"WARNING: could not read {eqt_volpick_h5}: {e}")
        return arr_out

    need = {"trace_id", "peak_time", "peak_value", "phase"}
    if not need.issubset(set(picks_df.columns)):
        print(f"WARNING: eqt-volpick picks table missing columns {need - set(picks_df.columns)}")
        return arr_out

    picks_df = picks_df.copy()
    picks_df["peak_time"] = pd.to_datetime(picks_df["peak_time"], utc=True)
    picks_df["_sta"] = picks_df["trace_id"].astype(str).map(_station_code_from_id)
    picks_df["_fam"] = picks_df["phase"].map(_phase_family)
    picks_df = picks_df[picks_df["_fam"].notna()]

    if t1 is not None and t2 is not None:
        t_lo = pd.Timestamp(t1, tz="UTC")
        t_hi = pd.Timestamp(t2, tz="UTC")
        picks_df = picks_df[(picks_df["peak_time"] >= t_lo) & (picks_df["peak_time"] <= t_hi)]

    arr_out["_sta"] = arr_out["station_id"].map(_station_code_from_id)

    probs: list[float] = []
    for _, r in arr_out.iterrows():
        fam = _phase_family(r.get("phase"))
        sta = r["_sta"]
        if fam is None:
            probs.append(np.nan)
            continue
        arr_t = pd.Timestamp(r["arrival_time"])
        if pd.isna(arr_t):
            probs.append(np.nan)
            continue
        if arr_t.tzinfo is None:
            arr_t = arr_t.tz_localize("UTC")
        else:
            arr_t = arr_t.tz_convert("UTC")

        sub = picks_df[(picks_df["_sta"] == sta) & (picks_df["_fam"] == fam)]
        if sub.empty:
            probs.append(np.nan)
            continue
        delta_s = (sub["peak_time"] - arr_t).dt.total_seconds().abs()
        idx_best = delta_s.idxmin()
        if float(delta_s.loc[idx_best]) <= EQT_PICK_MATCH_MAX_DT_S:
            probs.append(float(sub.loc[idx_best, "peak_value"]))
        else:
            probs.append(np.nan)

    arr_out["prob"] = probs
    matched = arr_out["prob"].notna().sum()
    print(
        f"  Pick probability join (eqt-volpick): {matched}/{len(arr_out)} arrivals matched "
        f"(nearest peak within {EQT_PICK_MATCH_MAX_DT_S:g}s, same station & P/S)"
    )
    return arr_out


def make_catalog_volcano_figure(
    cfg: dict,
    cat_df: pd.DataFrame,
    *,
    inventory_path: Path,
    map_t1: str | None,
    map_t2: str | None,
    out_png: Path | None = None,
) -> Path | VolcanoFigure:
    """Overview map + cross-sections via :class:`~vdapseisutils.VolcanoFigure`.

    If ``out_png`` is set, saves to disk, closes the figure, and returns the path.
    Otherwise returns the figure for display (e.g. in Jupyter).
    """
    origin = (float(cfg["target"]["latitude"]), float(cfg["target"]["longitude"]))
    inventory = read_inventory(str(inventory_path))

    obspy_cat = _cat_df_to_obspy(cat_df)

    fig = VolcanoFigure(
        origin=origin,
        radial_extent_km=15.0,
        depth_extent=(-15.0, 5.0),
    )
    # Use Map.add_terrain (explicit zoom/cache only) so VolcanoFigure cannot forward
    # stray kwargs (e.g. style=) into add_arcgis_terrain on older vdapseisutils builds.
    try:
        fig.map_obj.add_terrain()
    except Exception as e:
        print(f"Terrain tiles failed: {e}")
    try:
        fig.add_hillshade(alpha=0.6)
    except Exception as e:
        print(f"Hillshade failed: {e}")

    fig.plot_catalog(
        obspy_cat,
        s=25,
        c=NLL_CATALOG_COLOR,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.4,
    )
    fig.plot_inventory(inventory, s=35, c="black", alpha=0.9, cross_section_s=12)

    win = (
        f"{map_t1} to {map_t2}"
        if map_t1 is not None and map_t2 is not None
        else "full catalog"
    )
    try:
        fig.map_obj.ax.set_title(
            f"{cfg['target']['name']} located earthquakes | {win} | N={len(cat_df)}",
            fontsize=10,
        )
    except Exception:
        pass

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_png), dpi=250)
        plt.close(fig)
        return out_png
    return fig


def make_map(cfg: dict, cat_df: pd.DataFrame, *, paths: StHelensVizPaths, t1: str, t2: str, out_dir: Path) -> Path:
    """Backward-compatible name: write ``sthelens_map.png`` under ``out_dir``."""
    out_png = Path(out_dir) / "sthelens_map.png"
    p = make_catalog_volcano_figure(
        cfg,
        cat_df,
        inventory_path=paths.inventory_path,
        map_t1=t1,
        map_t2=t2,
        out_png=out_png,
    )
    assert p is not None
    return p


def _cat_df_to_obspy(cat_df: pd.DataFrame) -> Catalog:
    obspy_cat = Catalog()
    for _, r in cat_df.iterrows():
        ev = Event()
        ev_origin = Origin(
            time=UTCDateTime(pd.Timestamp(r["origin_time"]).to_pydatetime()),
            latitude=float(r["latitude"]),
            longitude=float(r["longitude"]),
            depth=float(r["depth_km"]) * 1000.0,
        )
        mag_val = pd.to_numeric(r.get("mag"), errors="coerce")
        if pd.isna(mag_val):
            mag_val = 1.0
        mag = Magnitude(mag=float(mag_val), magnitude_type="Md")
        ev.origins = [ev_origin]
        ev.magnitudes = [mag]
        ev.preferred_origin_id = ev_origin.resource_id
        ev.preferred_magnitude_id = mag.resource_id
        obspy_cat.append(ev)
    return obspy_cat


def _load_event_stream(
    sds_client: SDSClient,
    network: str,
    stations: list[str],
    t_start: UTCDateTime,
    t_end: UTCDateTime,
) -> Stream:
    st = Stream()
    for sta in stations:
        try:
            st_sta = sds_client.get_waveforms(network, sta, "*", "?HZ", t_start, t_end)
            if len(st_sta) == 0:
                continue
            st_sta = st_sta.merge(fill_value=0)
            for tr in st_sta:
                tr.detrend("demean")
                tr.filter("bandpass", freqmin=BANDPASS[0], freqmax=BANDPASS[1], corners=4, zerophase=True)
            st += st_sta
        except Exception as e:
            print(f"    skip {network}.{sta}: {e}")
    st.sort(keys=["station"])
    return st


def _nll_file_for_origin_time(
    otime: UTCDateTime, suffix: str, *, nll_loc_root: Path, nll_run_prefix: str
) -> Path | None:
    """Locate the NLL per-event output file for a given origin time.

    NLL writes files like
    ``.../sthelens/loc-YYYY-MM-DD/{prefix}.YYYYMMDD.HHMMSS.grid0.loc.<suffix>``.
    The timestamp in the filename is rounded down to the second; occasionally
    it can differ by 1s from the reported origin time. We look for an exact
    match first, then fall back to the nearest file within +/- 30s.
    """
    day_dir = Path(nll_loc_root) / f"loc-{otime.datetime.strftime('%Y-%m-%d')}"
    if not day_dir.is_dir():
        return None

    stem = otime.datetime.strftime("%Y%m%d.%H%M%S")
    exact = day_dir / f"{nll_run_prefix}.{stem}.grid0.loc.{suffix}"
    if exact.is_file():
        return exact

    candidates = list(day_dir.glob(f"{nll_run_prefix}.*.grid0.loc.{suffix}"))
    best = None
    best_dt = 60.0
    for c in candidates:
        try:
            parts = c.stem.split(".")
            ts = UTCDateTime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
            dt = abs(ts - otime)
            if dt < best_dt:
                best_dt = dt
                best = c
        except Exception:
            continue
    if best is not None and best_dt <= 30.0:
        return best
    return None


def _pick_alpha(prob: float | None) -> float:
    """Map pick probability to line alpha, with a floor so weak picks stay visible."""
    if prob is None or not np.isfinite(prob):
        return 0.4
    return float(max(0.15, min(1.0, prob)))


def _pick_line_color(phase) -> str:
    """P-type arrivals → red, S-type → blue (matches common practice); unknown → gray."""
    p = str(phase).strip().upper()
    if not p:
        return "gray"
    if p.startswith("P"):
        return "red"
    if p.startswith("S"):
        return "blue"
    return "gray"


def _phase_family(phase) -> str | None:
    """Map phase label to 'P' or 'S' for matching picker rows; else None."""
    p = str(phase).strip().upper()
    if not p:
        return None
    if p.startswith("P"):
        return "P"
    if p.startswith("S"):
        return "S"
    return None


def _decode_id_cell(val) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace").strip()
    if isinstance(val, float) and pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def _format_arrival_timestamp_utc(ts) -> str:
    """UTC timestamp string for an NLL arrival (ISO-8601 with ms)."""
    t = pd.Timestamp(ts)
    if pd.isna(t):
        return ""
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    base = t.strftime("%Y-%m-%dT%H:%M:%S")
    us = t.microsecond
    if us:
        frac = f"{us:06d}".rstrip("0")
        return f"{base}.{frac}Z" if frac else f"{base}Z"
    return f"{base}Z"


def _phase_hint_ps(phase) -> str:
    fam = _phase_family(phase)
    if fam in ("P", "S"):
        return fam
    p = str(phase).strip().upper()
    return p if p else "?"


def _wave_id_from_arrival_row(r: pd.Series, *, default_network: str) -> str:
    """SeedLink-style id: network.station.location.channel when trace_id is present."""
    tid = _decode_id_cell(r.get("trace_id"))
    if tid:
        return tid
    sid = _decode_id_cell(r.get("station_id"))
    parts = sid.split(".")
    parts = [p for p in parts if p]
    if len(parts) >= 4:
        return ".".join(parts[:4])
    sta = _decode_id_cell(r.get("_sta")) or _station_code_from_id(sid)
    if default_network and sta:
        return f"{default_network}.{sta}.."
    return sid or "?"


def _build_location_picks_table_figure(
    ev_arr: pd.DataFrame, *, default_network: str
) -> plt.Figure:
    """Tabular listing of picks used in the locator: time, wave id, P|S, eqt-volpick prob."""
    headers = ["Timestamp", "wave_id", "phase_hint", "peak_probability"]
    if ev_arr is None or len(ev_arr) == 0:
        fig = plt.figure(figsize=(10, 1.0), dpi=180)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.05, 0.25, 0.9, 0.55])
        ax.axis("off")
        ax.text(
            0.5, 0.5, "No arrivals for this event.",
            ha="center", va="center", fontsize=8,
        )
        fig.suptitle(
            "Picks in location (NLL arrivals + eqt-volpick peak_probability)",
            fontsize=8, y=0.92,
        )
        return fig

    sub = ev_arr.sort_values("arrival_time", kind="mergesort")
    rows: list[list[str]] = []
    n = len(sub)
    fs = 5 if n > 30 else 6 if n > 18 else 7
    for _, r in sub.iterrows():
        prob = r.get("prob", np.nan)
        rows.append(
            [
                _format_arrival_timestamp_utc(r.get("arrival_time")),
                _wave_id_from_arrival_row(r, default_network=default_network),
                _phase_hint_ps(r.get("phase")),
                f"{float(prob):.4f}" if np.isfinite(prob) else "NA",
            ]
        )

    fig_h = min(0.38 * (2 + n), 22.0)
    fig = plt.figure(figsize=(10.0, fig_h), dpi=180)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.86])
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=[0.28, 0.38, 0.10, 0.20],
        cellLoc="left",
        loc="upper center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fs)
    tbl.scale(1.0, 1.22)
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
    fig.suptitle(
        "Picks in location — NLL arrival time, waveform id, phase_hint, eqt-volpick peak_probability",
        fontsize=8,
        y=0.98,
    )
    return fig


def _build_pick_alpha_legend_figure() -> plt.Figure:
    """Small standalone figure: two horizontal colorbars for P/S line α vs eqt-volpick probability."""
    fig = plt.figure(figsize=(10, 1.15), dpi=180)
    fig.patch.set_facecolor("white")
    n = 256
    probs = np.linspace(0.0, 1.0, n)
    alphas = np.array([_pick_alpha(float(p)) for p in probs])
    rgba_p = np.zeros((n, 4))
    rgba_p[:, 0] = 1.0
    rgba_p[:, 3] = alphas
    rgba_s = np.zeros((n, 4))
    rgba_s[:, 2] = 1.0
    rgba_s[:, 3] = alphas
    cmap_p = ListedColormap(rgba_p)
    cmap_s = ListedColormap(rgba_s)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    sm_p = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_p)
    sm_p.set_array([])
    sm_s = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_s)
    sm_s.set_array([])
    cax_p = fig.add_axes([0.10, 0.52, 0.80, 0.22])
    cax_s = fig.add_axes([0.10, 0.14, 0.80, 0.22])
    cb_p = fig.colorbar(sm_p, cax=cax_p, orientation="horizontal")
    cb_s = fig.colorbar(sm_s, cax=cax_s, orientation="horizontal")
    cb_p.set_label("P pick line α (eqt-volpick pick probability)", fontsize=7, labelpad=2)
    cb_s.set_label("S pick line α (eqt-volpick pick probability)", fontsize=7, labelpad=2)
    cb_p.ax.tick_params(labelsize=6)
    cb_s.ax.tick_params(labelsize=6)
    fig.suptitle(
        "Pick line opacity (α) vs eqt-volpick pick probability",
        fontsize=8,
        y=0.98,
    )
    return fig


def _figure_to_pil(fig: plt.Figure, dpi: int, *, pad_inches: float = 0.12) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _stack_images_vertical(im_top: Image.Image, im_bottom: Image.Image) -> Image.Image:
    w = max(im_top.width, im_bottom.width)

    def _fit_width(im: Image.Image) -> Image.Image:
        if im.width == w:
            return im
        nh = max(1, int(im.height * w / im.width))
        return im.resize((w, nh), Image.Resampling.LANCZOS)

    t = _fit_width(im_top)
    b = _fit_width(im_bottom)
    out = Image.new("RGB", (w, t.height + b.height), (255, 255, 255))
    out.paste(t, (0, 0))
    out.paste(b, (0, t.height))
    return out


def _stack_images_vertical_list(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("no images")
    out = images[0]
    for im in images[1:]:
        out = _stack_images_vertical(out, im)
    return out


def _build_event_map_xs_figure(
    cfg: dict,
    inventory,
    cat_df: pd.DataFrame,
    event_idx: int,
    n_events: int,
    paths: StHelensVizPaths,
) -> tuple[plt.Figure, int]:
    """Map (left) + two cross-sections (right), VolcanoFigure-like but no time series."""
    row = cat_df.iloc[event_idx]
    otime = UTCDateTime(pd.Timestamp(row["origin_time"]).to_pydatetime())

    hyp_path = _nll_file_for_origin_time(
        otime, "hyp", nll_loc_root=paths.nll_loc_root, nll_run_prefix=paths.nll_run_prefix
    )
    scat_path = _nll_file_for_origin_time(
        otime, "scat", nll_loc_root=paths.nll_loc_root, nll_run_prefix=paths.nll_run_prefix
    )

    origin = (
        float(cfg["target"]["latitude"]),
        float(cfg["target"]["longitude"]),
    )
    fig = plt.figure(figsize=(11.5, 6.8), dpi=200)
    spec = fig.add_gridspec(
        2,
        2,
        # Right column (both cross-sections): ~1.25× width of map column
        width_ratios=[1.0, 1.25],
        height_ratios=[1.0, 1.0],
        hspace=0.2,
        wspace=0.14,
        top=0.93,
        bottom=0.06,
        left=0.05,
        right=0.95,
    )
    fig_m = fig.add_subfigure(spec[0:2, 0])
    fig_xs1 = fig.add_subfigure(spec[0, 1])
    fig_xs2 = fig.add_subfigure(spec[1, 1])

    map_obj = Map(fig=fig_m, origin=origin, radial_extent_km=EVENT_MAP_RADIUS_KM)
    xs1 = CrossSection(
        fig=fig_xs1,
        origin=origin,
        azimuth=270.0,
        radius_km=EVENT_MAP_RADIUS_KM,
        depth_extent=EVENT_MAP_DEPTH_EXTENT,
        label="A",
    )
    xs2 = CrossSection(
        fig=fig_xs2,
        origin=origin,
        azimuth=0.0,
        radius_km=EVENT_MAP_RADIUS_KM,
        depth_extent=EVENT_MAP_DEPTH_EXTENT,
        label="B",
    )
    # Hillshade first so section lines and data draw above terrain
    try:
        map_obj.add_hillshade(alpha=0.6)
    except Exception as e:
        print(f"    hillshade failed: {e}")
    map_obj.plot_line(
        xs1.properties["points"][0],
        xs1.properties["points"][1],
        label="A",
    )
    map_obj.plot_line(
        xs2.properties["points"][0],
        xs2.properties["points"][1],
        label="B",
    )
    map_obj.add_scalebar()

    n_scat = 0
    if scat_path is not None and hyp_path is not None:
        try:
            xyz = nllpy.read_scat(scat_path)
            n_scat = int(xyz.shape[0])
            lat_o, lon_o, rot = nllpy.parse_hyp_transform(hyp_path)
            s_lat, s_lon, s_depth = nllpy.scat_to_geographic(xyz, lat_o, lon_o, rot)
            map_obj.ax.scatter(
                s_lon,
                s_lat,
                s=2.0,
                c=SCATTER_CLOUD_TEAL,
                alpha=0.12,
                transform=ccrs.Geodetic(),
                zorder=1,
            )
            for xs in (xs1, xs2):
                xs.scatter(
                    lat=s_lat,
                    lon=s_lon,
                    z=s_depth,
                    z_dir="depth",
                    z_unit="km",
                    s=1.5,
                    c=SCATTER_CLOUD_TEAL,
                    alpha=0.08,
                    zorder=1,
                )
        except Exception as e:
            print(f"    scatter cloud failed: {e}")

    ev_cat = _cat_df_to_obspy(cat_df.iloc[[event_idx]])
    map_obj.plot_catalog(
        ev_cat,
        s=60,
        c=NLL_CATALOG_COLOR,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )
    for xs in (xs1, xs2):
        xs.plot_catalog(
            ev_cat,
            s=60,
            c=NLL_CATALOG_COLOR,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
    map_obj.plot_inventory(inventory, s=45, c="black", alpha=0.9, zorder=6)
    for xs in (xs1, xs2):
        xs.plot_inventory(inventory, s=14, c="black", alpha=0.9, zorder=6)

    try:
        map_obj.ax.set_title(
            f"Event {event_idx+1:02d}/{n_events} | {otime.datetime.strftime('%Y-%m-%d %H:%M:%S')}Z | "
            f"epi lat={row['latitude']:.4f} lon={row['longitude']:.4f} "
            f"depth={row['depth_km']:.2f} km | "
            f"rms={row['residual_rms']:.3f}s | N_scat={n_scat} | "
            f"map {EVENT_MAP_RADIUS_KM:.0f} km on {cfg['target']['name']} | "
            f"A–A′ E–W, B–B′ N–S",
            fontsize=9,
        )
    except Exception:
        pass

    return fig, n_scat


def _build_clipboard_figure(
    cfg: dict,
    row: pd.Series,
    ev_arr: pd.DataFrame,
    event_idx: int,
    n_events: int,
    st: Stream,
) -> plt.Figure:
    """Waveform clipboard with picks and caption (P/S α legend is a separate figure)."""
    otime = UTCDateTime(pd.Timestamp(row["origin_time"]).to_pydatetime())
    t1 = otime - WAVEFORM_PAD_BEFORE_S
    t2 = otime + WAVEFORM_PAD_AFTER_S

    cb = plt.figure(
        FigureClass=ClipboardClass,
        st=st,
        mode="w",
        figsize=(10, max(4.5, 0.88 * len(st))),
        tick_type="datetime",
        sync_waves=True,
    )
    cb.plot()
    cb.set_tlim((t1.datetime, t2.datetime))

    top_ax = cb.get_axes()[0]
    top_ax.tick_params(axis="x", which="both", top=False, labeltop=False)
    top_ax.xaxis.tick_bottom()
    top_ax.set_xticklabels([])

    cb.axvline([otime.datetime], color="black", linestyle="--", linewidth=0.9, alpha=0.7)

    for i_tr, tr in enumerate(cb.st):
        sta = tr.stats.station
        sta_arr = ev_arr[ev_arr["_sta"] == sta]
        if sta_arr.empty:
            continue
        sf = cb.subfigs[i_tr]
        tlim = cb.taxis["time_lim"][i_tr]
        for ax in sf.axes:
            xlim = ax.get_xlim()
            for _, a in sta_arr.iterrows():
                dt = pd.Timestamp(a["arrival_time"]).to_pydatetime()
                prob = a.get("prob", np.nan)
                alpha = _pick_alpha(prob)
                color = _pick_line_color(a["phase"])
                lw = 2.0 if color == "blue" else 1.0
                x = t2axiscoords([dt], tlim, xlim, unit="datetime")[0]
                ax.axvline(x, color=color, linestyle="-", linewidth=lw, alpha=alpha)
                label = f"{float(prob):.2f}" if np.isfinite(prob) else "NA"
                ax.text(
                    x, 1.02, label,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom",
                    fontsize=6, color=color, alpha=min(1.0, alpha + 0.15),
                    clip_on=False,
                )

    cb.subplots_adjust(
        top=CLIPBOARD_SUBPLOTS_TOP,
        bottom=CLIPBOARD_SUBPLOTS_BOTTOM,
    )
    cb.suptitle(
        f"St. Helens | Event {event_idx+1:02d}/{n_events} | "
        f"{otime.datetime.strftime('%Y-%m-%d %H:%M:%S')}Z | "
        f"lat={row['latitude']:.4f} lon={row['longitude']:.4f} "
        f"depth={row['depth_km']:.2f} km | rms={row['residual_rms']:.3f}s",
        fontsize=9,
        y=CLIPBOARD_SUPTITLE_Y,
    )
    cb.text(
        0.5,
        CLIPBOARD_CAPTION_Y,
        "Black dashed = origin time. Numbers above pick lines = eqt-volpick pick probability "
        "(missing/invalid → line α=0.40). Below: P/S α legend, then tabular list of location picks.",
        ha="center",
        va="bottom",
        fontsize=7,
        style="italic",
        color="#444444",
        transform=cb.transFigure,
    )
    return cb


def make_event_map_and_clipboard_combined(
    cfg: dict,
    cat_df: pd.DataFrame,
    arr_df: pd.DataFrame,
    *,
    paths: StHelensVizPaths,
    clipboard_dir: Path,
) -> list[Path]:
    """For each event: map + XS | waveforms | P/S α legend | picks table in one PNG."""
    clipboard_dir = Path(clipboard_dir)
    clipboard_dir.mkdir(parents=True, exist_ok=True)
    for stale in clipboard_dir.glob("*-map.png"):
        try:
            stale.unlink()
        except OSError:
            pass

    inventory = read_inventory(str(paths.inventory_path))
    network_code = inventory[0].code
    sds_client = SDSClient(cfg["waveform_client"]["datasource"])

    cat_df = cat_df.reset_index(drop=True)
    written: list[Path] = []
    n = len(cat_df)
    for i in range(n):
        row = cat_df.iloc[i]
        ev_id = row["event_id"]
        short_id = ev_id.split("/")[-1][:8]
        otime = UTCDateTime(pd.Timestamp(row["origin_time"]).to_pydatetime())
        safe_time = otime.datetime.strftime("%Y%m%dT%H%M%S")

        ev_arr = arr_df[arr_df["event_id"] == ev_id].copy()
        if "_sta" not in ev_arr.columns:
            ev_arr["_sta"] = ev_arr["station_id"].map(_station_code_from_id)
        stations_in_event = sorted(set(ev_arr["_sta"]))

        scat_path = _nll_file_for_origin_time(
            otime, "scat", nll_loc_root=paths.nll_loc_root, nll_run_prefix=paths.nll_run_prefix
        )
        print(
            f"[{i+1:02d}/{n}] {otime.datetime.isoformat()} id={short_id} "
            f"stations={len(stations_in_event)} arrivals={len(ev_arr)} "
            f"scat={'ok' if scat_path else 'MISSING'}"
        )

        mfig, _n_scat = _build_event_map_xs_figure(cfg, inventory, cat_df, i, n, paths)

        t1 = otime - WAVEFORM_PAD_BEFORE_S
        t2 = otime + WAVEFORM_PAD_AFTER_S
        st = _load_event_stream(sds_client, network_code, stations_in_event, t1, t2)
        if len(st) == 0:
            print("    no waveforms available, skipping")
            plt.close(mfig)
            continue

        cb_fig = _build_clipboard_figure(cfg, row, ev_arr, i, n, st)
        leg_fig = _build_pick_alpha_legend_figure()
        tab_fig = _build_location_picks_table_figure(ev_arr, default_network=network_code)

        try:
            im_map = _figure_to_pil(mfig, dpi=200, pad_inches=0.08)
            im_cb = _figure_to_pil(cb_fig, dpi=180, pad_inches=0.12)
            im_leg = _figure_to_pil(leg_fig, dpi=180, pad_inches=0.08)
            im_tab = _figure_to_pil(tab_fig, dpi=180, pad_inches=0.1)
            combined = _stack_images_vertical_list([im_map, im_cb, im_leg, im_tab])
            out_png = clipboard_dir / f"event_{i+1:02d}_{safe_time}_{short_id}.png"
            combined.save(str(out_png))
            written.append(out_png)
        finally:
            plt.close(mfig)
            plt.close(cb_fig)
            plt.close(leg_fig)
            plt.close(tab_fig)

    return written


def render_single_event_combined_pil(
    cfg: dict,
    paths: StHelensVizPaths,
    cat_df: pd.DataFrame,
    arr_df: pd.DataFrame,
    *,
    event_id: str,
) -> Image.Image | None:
    """
    In-memory stack (map + XS | clipboard | P/S α legend | picks table) for one event.

    Returns ``None`` if ``event_id`` is missing or waveforms are unavailable for that event.
    """
    cat_df = cat_df.reset_index(drop=True)
    match = cat_df["event_id"].astype(str) == str(event_id)
    if match.sum() != 1:
        return None
    event_idx = int(np.flatnonzero(match.to_numpy())[0])
    n_events = len(cat_df)
    row = cat_df.iloc[event_idx]
    ev_id = row["event_id"]
    otime = UTCDateTime(pd.Timestamp(row["origin_time"]).to_pydatetime())

    inventory = read_inventory(str(paths.inventory_path))
    network_code = inventory[0].code
    sds_client = SDSClient(cfg["waveform_client"]["datasource"])

    ev_arr = arr_df[arr_df["event_id"] == ev_id].copy()
    if "_sta" not in ev_arr.columns:
        ev_arr["_sta"] = ev_arr["station_id"].map(_station_code_from_id)
    stations_in_event = sorted(set(ev_arr["_sta"]))

    mfig, _n_scat = _build_event_map_xs_figure(cfg, inventory, cat_df, event_idx, n_events, paths)
    t1 = otime - WAVEFORM_PAD_BEFORE_S
    t2 = otime + WAVEFORM_PAD_AFTER_S
    st = _load_event_stream(sds_client, network_code, stations_in_event, t1, t2)
    if len(st) == 0:
        plt.close(mfig)
        return None

    cb_fig = _build_clipboard_figure(cfg, row, ev_arr, event_idx, n_events, st)
    leg_fig = _build_pick_alpha_legend_figure()
    tab_fig = _build_location_picks_table_figure(ev_arr, default_network=network_code)
    try:
        im_map = _figure_to_pil(mfig, dpi=200, pad_inches=0.08)
        im_cb = _figure_to_pil(cb_fig, dpi=180, pad_inches=0.12)
        im_leg = _figure_to_pil(leg_fig, dpi=180, pad_inches=0.08)
        im_tab = _figure_to_pil(tab_fig, dpi=180, pad_inches=0.1)
        return _stack_images_vertical_list([im_map, im_cb, im_leg, im_tab])
    finally:
        plt.close(mfig)
        plt.close(cb_fig)
        plt.close(leg_fig)
        plt.close(tab_fig)


def _wav2hyp_repo_root() -> Path:
    """``wav2hyp`` repository root (contains ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[3]


def main() -> None:
    """CLI entry: legacy ``results_local/sthelens`` layout + ``analysis_local/`` output."""
    repo = _wav2hyp_repo_root()
    config_path = repo / "examples_local" / "sthelens.yaml"
    if not config_path.is_file():
        config_path = repo / "examples" / "sthelens.yaml"
    run_base = repo / "results_local" / "sthelens"
    out_dir = repo / "analysis_local" / "sthelens_20040923_22-24utc"
    clipboard_dir = out_dir / "clipboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    clipboard_dir.mkdir(parents=True, exist_ok=True)

    nll_t1 = "2004-09-23T22:00:00"
    nll_t2 = "2004-09-24T00:00:00"

    cfg = load_config_from_path(config_path)
    paths = sthelens_paths_with_optional_legacy_subdirs(cfg, config_path=config_path, run_base_dir=run_base)

    print(f"Reading catalog & arrivals from nll.h5 for {nll_t1} .. {nll_t2}")
    cat_df, arr_df = load_catalog_and_arrivals(paths.nll_h5, nll_t1, nll_t2)
    print(f"  Found {len(cat_df)} located events, {len(arr_df)} arrivals")

    print("Attaching pick probabilities from eqt-volpick ...")
    arr_df = attach_pick_probabilities(
        cat_df, arr_df, eqt_volpick_h5=paths.eqt_volpick_h5, t1=nll_t1, t2=nll_t2
    )

    print("Producing overview map ...")
    map_png = make_map(cfg, cat_df, paths=paths, t1=nll_t1, t2=nll_t2, out_dir=out_dir)
    print(f"  Saved {map_png}")

    print("Producing per-event map + clipboard PNGs ...")
    combined_pngs = make_event_map_and_clipboard_combined(
        cfg, cat_df, arr_df, paths=paths, clipboard_dir=clipboard_dir
    )
    print(f"  Wrote {len(combined_pngs)} combined figures under {clipboard_dir}")


if __name__ == "__main__":
    main()
