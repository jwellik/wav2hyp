"""
Shared colors and Matplotlib axis styling for wav2hyp maps and analysis notebooks.

Aligned with ``analysis_local/utils/sthelens_plot_styles.py`` (80s palette;
NLL / located catalog uses the first swatch).
"""

from __future__ import annotations

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

# arrival diagnostics (match reference boxplot convention)
P_PHASE_COMPARER_COLOR = "#C62828"
S_PHASE_COMPARER_COLOR = "#1565C0"

# diverging colormap name for depth difference (test − canonical), km
DELTA_DEPTH_CMAP = "coolwarm"


def apply_mpl_axes_style(ax, facecolor: str = "white", grid_alpha: float = 0.35) -> None:
    """Apply consistent Matplotlib axis styling (matches analysis notebooks)."""
    ax.set_facecolor(facecolor)
    ax.grid(True, alpha=grid_alpha)
