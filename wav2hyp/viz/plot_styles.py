"""
Shared colors and Matplotlib axis styling for wav2hyp maps and analysis notebooks.

Aligned with ``analysis_local/utils/sthelens_plot_styles.py`` (80s palette;
NLL / located catalog uses the first swatch).
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# 80s-inspired palette (purple first) — keep in sync with sthelens_plot_styles.COLORS_80S
COLORS_80S = [
    "#8A2BE2",  # neon purple — NLL / located earthquakes, event-rate curves
    "#FF5EA8",  # hot pink
    "#00E5FF",  # cyan
    "#39FF14",  # neon green
    "#FFA500",  # orange
    "#FFD700",  # yellow
    "#00BFFF",  # electric blue
    "#FF1493",  # deep pink
]

NLL_CATALOG_COLOR = COLORS_80S[0]
# Teal scatter cloud vs purple hypocenters / NLL solutions
SCATTER_CLOUD_TEAL = "#14B8A6"

# catalog_comparer: three meta-catalog map layers (distinct from the default NLL purple)
META_CANONICAL_ONLY_COLOR = COLORS_80S[3]  # green
META_TEST_ONLY_COLOR = COLORS_80S[4]  # orange
META_BOTH_COLOR = COLORS_80S[1]  # hot pink

# arrival diagnostics: 80s-style P/S (for catalog comparer boxplots)
P_PHASE_80S = COLORS_80S[1]  # hot pink
S_PHASE_80S = COLORS_80S[2]  # cyan
# legacy names (darker) — prefer P_PHASE_80S / S_PHASE_80S in new comparer figures
P_PHASE_COMPARER_COLOR = "#C62828"
S_PHASE_COMPARER_COLOR = "#1565C0"

# diverging colormap for depth difference (test − canonical), km
DELTA_DEPTH_CMAP = "coolwarm"
# 80s diverging, symmetric "heat": neon cyan (negative) and hot pink (positive) with
# equal vividness; neither side reads as the single "warm" pole (unlike red vs blue).
DELTA_DEPTH_CMAP_80S = LinearSegmentedColormap.from_list(
    "wav2hyp_delta_depth_80s",
    [COLORS_80S[2], "#E8E8E8", COLORS_80S[1]],
    N=256,
)
# Default symmetric colorbar half-range (km) for azimuth–distance / Δ depth
DELTA_DEPTH_COLORBAR_HALFRANGE_KM = 50.0

# Shared x-limits for arrival Δt diagnostics (seconds)
ARRIVAL_DELTA_T_XLIM: tuple[float, float] = (-10.0, 10.0)
# Boxplot-by-station: tighter window than scatter
ARRIVAL_BOXPLOT_DELTA_T_XLIM: tuple[float, float] = (-2.5, 2.5)


def apply_mpl_axes_style(ax, facecolor: str = "white", grid_alpha: float = 0.35) -> None:
    """Apply consistent Matplotlib axis styling (matches analysis notebooks)."""
    ax.set_facecolor(facecolor)
    ax.grid(True, alpha=grid_alpha)
