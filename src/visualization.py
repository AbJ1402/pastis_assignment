"""Plotly styling, class colours and Sentinel-2 image helpers shared by the notebook and scripts."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from src.data_loading import CLASS_NAMES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# Categorical order validated for colour-vision deficiency; use slots in order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Official PASTIS legend from the assignment brief (tab20 with black background and white void).
PASTIS_COLORS = [
    "#000000", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
    "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
    "#bcbd22", "#dbdb8d", "#17becf", "#ffffff",
]

TRUE_COLOR = (2, 1, 0)          # B4, B3, B2
COLOR_INFRARED = (6, 2, 1)      # B8, B4, B3: photosynthetically active vegetation shows red
SWIR_AGRICULTURE = (8, 6, 0)    # B11, B8, B2: green crops vs brown/pink bare soil


def register_template(name: str = "pastis") -> None:
    axis = dict(
        gridcolor=GRIDLINE, linecolor=BASELINE, zerolinecolor=BASELINE, showline=True,
        tickfont=dict(color=INK_SECONDARY), title=dict(font=dict(color=INK_SECONDARY)),
    )
    pio.templates[name] = go.layout.Template(layout=dict(
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
        title=dict(font=dict(size=16, color=INK), x=0.01, xanchor="left"),
        xaxis=axis, yaxis=axis,
        legend=dict(font=dict(color=INK_SECONDARY)),
        hoverlabel=dict(font=dict(family=FONT)),
        barcornerradius=4, bargap=0.35,
    ))
    pio.templates.default = f"plotly_white+{name}"


def with_alpha(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def s2_composite(frame: np.ndarray, bands=TRUE_COLOR, clip_pct: float = 2,
                 per_channel: bool = False, value_range: tuple[float, float] | None = None) -> np.ndarray:
    """(10, H, W) single acquisition -> (H, W, 3) uint8.
    Default: percentile stretch (shared across channels keeps natural colour balance; per-channel
    suits false-colour composites). `value_range` (in DN) applies one fixed stretch instead, so
    brightness is comparable across dates."""
    img = frame[list(bands)].astype(np.float32).transpose(1, 2, 0)
    if value_range is not None:
        lo, hi = value_range
    else:
        axis = (0, 1) if per_channel else None
        lo = np.percentile(img, clip_pct, axis=axis)
        hi = np.percentile(img, 100 - clip_pct, axis=axis)
    return (np.clip((img - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def class_colorscale(colors: list[str] = PASTIS_COLORS) -> list:
    """Stepped colourscale: class k occupies [k - 0.5, k + 0.5] when zmin=-0.5, zmax=n-0.5."""
    n = len(colors)
    scale = []
    for k, color in enumerate(colors):
        scale += [[k / n, color], [(k + 1) / n, color]]
    return scale


def label_heatmap(label: np.ndarray, showscale: bool = False) -> go.Heatmap:
    n = len(PASTIS_COLORS)
    return go.Heatmap(
        z=label, zmin=-0.5, zmax=n - 0.5, colorscale=class_colorscale(), showscale=showscale,
        colorbar=dict(tickvals=list(range(n)), ticktext=[CLASS_NAMES[k] for k in range(n)],
                      len=1.0, outlinewidth=1, outlinecolor=BASELINE, tickfont=dict(size=11)),
        hovertemplate="row %{y}, col %{x}<br>class %{z}<extra></extra>",
    )


def style_image_axes(fig: go.Figure, x_title: str = "Column (px)", reverse_y: bool = False) -> go.Figure:
    """Square pixels, no ticks or grid, and an x-axis title under every image panel."""
    fig.for_each_xaxis(lambda ax: ax.update(
        showticklabels=False, showgrid=False, showline=False, zeroline=False, title_text=x_title))
    fig.for_each_yaxis(lambda ax: ax.update(
        showticklabels=False, showgrid=False, showline=False, zeroline=False, scaleanchor=ax.anchor,
        autorange="reversed" if reverse_y else ax.autorange))
    return fig
