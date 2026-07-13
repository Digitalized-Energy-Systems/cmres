"""Publication figure style for the CMRES evaluation bar plots.

A faithful port of the style/template used in
``C:\\Users\\Public\\git\\scare\\experiment\\eval\\plots.py`` — same theme,
bar outline, colour-blind hatch overlay, top legend strip, and compact
horizontal-bar sizing — adapted for the cmres eval pipeline.

The one intentional deviation from scare: the **sector / carrier palette
keeps the cmres traditional hues** (electricity = amber, heat = red,
gas = green) rather than scare's blue/green/red, because the dissertation's
network figures already use these and we keep sector colours where they
carry meaning. Non-sector categorical series (metrics, scenarios, …) use a
publication qualitative palette chosen to *not* clash with the three sector
hues, with a redundant hatch channel layered on top for CVD readers.

Used by both ``cmres_eval_plots`` (E16 bars) and ``cp_cn_evaluation``
(cross-carrier / ρ / impact bars).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import plotly.graph_objects as go

import cmres.evaluation.evaluation as _eval


# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────

# Sector / carrier hues — KEPT from the cmres convention so sector bars match
# the network/topology figures pixel-for-pixel.
#   electricity = amber, heat = red, gas = green.
# ``multi`` (coupling points) + ``total`` round out the sector legend.
SECTOR_COLOR = {
    "electricity": _eval.NETWORK_COLOR_MAP["electricity"],  # #ffa000 amber
    "power":       _eval.NETWORK_COLOR_MAP["electricity"],  # alias
    "heat":        _eval.NETWORK_COLOR_MAP["heat"],         # #d32f2f red
    "gas":         _eval.NETWORK_COLOR_MAP["gas"],          # #388e3c green
    "multi":       "#7e57c2",                               # purple — coupling pts
    "total":       "#444444",                               # dark grey
    "ranked":      "#1565c0",                               # blue — carrier-rank pooled
}

# Colour-blind hatch per sector — the redundant channel layered on the kept
# hue so sectors stay separable in greyscale / for CVD readers.
SECTOR_PATTERN = {
    "electricity": "",
    "power":       "",
    "heat":        "\\",
    "gas":         "/",
    "multi":       "x",
    "total":       "",
    "ranked":      ".",
}

# Pretty sector labels (legend text).
SECTOR_PRETTY = {
    "electricity": "Electricity",
    "power":       "Electricity",
    "heat":        "Heat",
    "gas":         "Gas",
    "multi":       "Multi (CPs)",
    "total":       "Total (raw pooled)",
    "ranked":      "Overall (carrier-rank)",
}

# Qualitative palette for non-sector categoricals (metrics, scenarios, null
# models, variants…). Deliberately avoids amber / strong-red / strong-green
# so a metric bar never reads as a sector bar. Colour-blind-aware ordering.
QUAL_PALETTE = [
    "#1F4E96",  # deep blue
    "#9467BD",  # purple
    "#17BECF",  # teal / cyan
    "#8C564B",  # brown
    "#E377C2",  # pink
    "#7F7F7F",  # grey
    "#2B83BA",  # mid blue
    "#5E4FA2",  # indigo
    "#AEC7E8",  # light blue
    "#C5B0D5",  # light purple
    "#9EDAE5",  # light teal
    "#C49C94",  # light brown
]

# Colour-blind-safe hatch shapes cycled for multi-series non-sector bars.
PATTERN_SHAPES = ("", "/", "\\", "x", "-", "+", "|", ".")


def qual_color(i: int) -> str:
    return QUAL_PALETTE[i % len(QUAL_PALETTE)]


def qual_pattern(i: int) -> str:
    return PATTERN_SHAPES[i % len(PATTERN_SHAPES)]


# ─────────────────────────────────────────────────────────────────────────────
# Typography / layout constants (from scare plots.py)
# ─────────────────────────────────────────────────────────────────────────────

_FONT_FAMILY = "Libertinus Sans, Inter, -apple-system, Segoe UI, Roboto, sans-serif"
_TITLE_FONT_FAMILY = "Libertinus Sans, Inter, -apple-system, Segoe UI, Roboto, sans-serif"

_FIG_WIDTH = 1000
_FIG_HEIGHT = 440

# Compact width for bar charts (the full-width default is for trajectories).
BAR_FIG_WIDTH = 720

_BASE_FONT_SIZE = 17
_TITLE_FONT_SIZE = 22
_AXIS_TITLE_FONT_SIZE = 18
_TICK_FONT_SIZE = 16
_LEGEND_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 14

_GRID_COLOR = "#ECECEC"
AXIS_COLOR = "#1A1A1A"
MUTED_COLOR = "#666666"

# Dark edge on every bar so adjacent fills separate cleanly in print.
BAR_LINE_COLOR = "#2A2A2A"
BAR_LINE_WIDTH = 0.8

_AXIS_STYLE = dict(
    gridcolor=_GRID_COLOR,
    gridwidth=0.8,
    zeroline=False,
    showline=False,
    mirror=False,
    ticks="",
    ticklen=0,
    tickcolor=AXIS_COLOR,
    tickwidth=0.9,
    tickfont=dict(size=_TICK_FONT_SIZE),
    title=dict(font=dict(size=_AXIS_TITLE_FONT_SIZE), standoff=10),
    automargin=True,
)
_X_AXIS_STYLE = {**_AXIS_STYLE, "showgrid": False}
_Y_AXIS_STYLE = {**_AXIS_STYLE, "showgrid": True}

_DEFAULT_LAYOUT = dict(
    template="plotly_white",
    width=_FIG_WIDTH,
    height=_FIG_HEIGHT,
    font=dict(family=_FONT_FAMILY, size=_BASE_FONT_SIZE, color=AXIS_COLOR),
    title=dict(
        font=dict(family=_TITLE_FONT_FAMILY, size=_TITLE_FONT_SIZE, color=AXIS_COLOR),
        x=0.5,
        xanchor="center",
        y=0.96,
        yanchor="top",
        pad=dict(t=6, b=6),
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=84, r=160, t=72, b=72),
    legend=dict(
        bgcolor="rgba(255,255,255,0)",
        bordercolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=_LEGEND_FONT_SIZE),
        orientation="v",
        x=1.02,
        xanchor="left",
        y=1.0,
        yanchor="top",
        itemsizing="constant",
        tracegroupgap=4,
    ),
    xaxis=_X_AXIS_STYLE,
    yaxis=_Y_AXIS_STYLE,
    hoverlabel=dict(
        bgcolor="white",
        bordercolor=AXIS_COLOR,
        font=dict(family=_FONT_FAMILY, size=13),
    ),
    bargap=0.2,
    bargroupgap=0.06,
)


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def _visible_len(s: str) -> int:
    """Character count ignoring HTML tags (so a ``<span>`` subtitle or
    ``<br>`` doesn't inflate the measured title width)."""
    out, depth = 0, 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out += 1
    return out


def bar_pattern(shape: Optional[str], *, fg: str = "white") -> Optional[dict]:
    """Subtle hatch ``marker.pattern`` for a bar, or ``None`` for a solid
    fill when ``shape`` is empty/None."""
    if not shape:
        return None
    return dict(shape=shape, solidity=0.30, size=7, fgcolor=fg, fgopacity=0.55)


def bar_marker(
    color: str,
    *,
    pattern_shape: Optional[str] = None,
    pattern_fg: str = "white",
) -> dict:
    """Bar marker with the shared dark outline and an optional colour-blind
    hatch overlaid on the solid fill."""
    marker: dict[str, Any] = dict(
        color=color,
        line=dict(color=BAR_LINE_COLOR, width=BAR_LINE_WIDTH),
    )
    pattern = bar_pattern(pattern_shape, fg=pattern_fg)
    if pattern is not None:
        marker["pattern"] = pattern
    return marker


def sector_marker(sector: str) -> dict:
    """``marker=`` for a sector / carrier bar: kept cmres hue + CVD hatch."""
    return bar_marker(
        SECTOR_COLOR.get(sector, "#888888"),
        pattern_shape=SECTOR_PATTERN.get(sector, ""),
    )


def hbar_height(n_groups: int, n_series: int = 1) -> int:
    """Compact height for a horizontal bar figure: enough per-category room
    to keep bars slim, plus headroom for title + top legend + axis."""
    if n_series <= 1:
        per_row = 30
    else:
        per_row = 15 * n_series + 10
    return int(min(720, max(230, per_row * n_groups + 150)))


def vbar_width(n_groups: int, n_series: int = 1, *, base: int = 360) -> int:
    """Compact width for a vertical (column) bar figure so a few columns
    don't sprawl across the canvas."""
    per = 26 * max(1, n_series) + 18
    return int(min(_FIG_WIDTH, max(base, per * n_groups + 180)))


# ─────────────────────────────────────────────────────────────────────────────
# Theme application (ported from scare ``_apply_theme``)
# ─────────────────────────────────────────────────────────────────────────────


def apply_theme(
    fig: go.Figure,
    *,
    title: str,
    height: int = _FIG_HEIGHT,
    width: int = _FIG_WIDTH,
    font_bump: int = 0,
    legend_top: bool = False,
    no_legend: bool = False,
) -> go.Figure:
    """Apply the shared figure theme.

    ``font_bump`` adds the same delta (pt) to every text element.
    ``legend_top`` moves the legend to a horizontal strip above the plot
    (used for bar charts) and reclaims the right margin. ``no_legend``
    reclaims the right margin when no legend is drawn.
    """
    fig.update_layout(_DEFAULT_LAYOUT)
    fig.update_layout(
        title=dict(text=title, **_DEFAULT_LAYOUT["title"]),
        height=height,
        width=width,
    )
    if no_legend:
        fig.update_layout(margin=dict(r=50))
    # Broadcast the shared axis theme to every subplot axis (update_layout
    # only styles axis #1). Overlay (secondary) y-axes keep their grid off.
    for axis in fig.select_xaxes():
        axis.update(_X_AXIS_STYLE)
    for axis in fig.select_yaxes():
        if axis.overlaying:
            axis.update({**_Y_AXIS_STYLE, "showgrid": False})
        else:
            axis.update(_Y_AXIS_STYLE)
    if font_bump:
        fig.update_layout(
            font=dict(size=_BASE_FONT_SIZE + font_bump),
            title=dict(
                text=title,
                **{
                    **_DEFAULT_LAYOUT["title"],
                    "font": dict(
                        family=_TITLE_FONT_FAMILY,
                        size=_TITLE_FONT_SIZE + font_bump,
                        color=AXIS_COLOR,
                    ),
                },
            ),
            legend=dict(
                **{
                    **_DEFAULT_LAYOUT["legend"],
                    "font": dict(size=_LEGEND_FONT_SIZE + font_bump),
                },
            ),
        )
        fig.update_xaxes(
            tickfont=dict(size=_TICK_FONT_SIZE + font_bump),
            title=dict(font=dict(size=_AXIS_TITLE_FONT_SIZE + font_bump), standoff=8),
        )
        fig.update_yaxes(
            tickfont=dict(size=_TICK_FONT_SIZE + font_bump),
            title=dict(font=dict(size=_AXIS_TITLE_FONT_SIZE + font_bump), standoff=8),
        )
        for ann in fig.layout.annotations or ():
            new_size = (ann.font.size or _AXIS_TITLE_FONT_SIZE) + font_bump
            ann.font.size = new_size
    # Pin every category tick so plotly doesn't thin labels on short panels.
    bar_orients = {
        getattr(tr, "orientation", None) or "v"
        for tr in fig.data
        if getattr(tr, "type", None) == "bar"
    }
    if "h" in bar_orients:
        fig.update_yaxes(dtick=1, tick0=0)
    elif bar_orients:
        fig.update_xaxes(dtick=1, tick0=0)

    if legend_top:
        legend_font = _LEGEND_FONT_SIZE + font_bump
        title_font = _TITLE_FONT_SIZE + font_bump
        longest_line = max(
            (_visible_len(seg) for seg in str(title).split("<br>")), default=1
        )
        fit_font = int(width * 0.92 / max(1, longest_line) / 0.52)
        title_font = max(13, min(title_font, fit_font))
        labels = [
            str(tr.name)
            for tr in fig.data
            if getattr(tr, "showlegend", None) is not False
            and getattr(tr, "name", None) not in (None, "")
        ]
        avail = max(1.0, width - 84)
        char_px = 0.56 * legend_font
        rows, cur = 1, 0.0
        for lab in labels:
            item_w = 34 + len(lab) * char_px + 18
            if cur > 0 and cur + item_w > avail:
                rows += 1
                cur = item_w
            else:
                cur += item_w
        n_title_lines = 1 + str(title).count("<br>")
        title_off = title_font * 1.15
        title_px = int(title_off + n_title_lines * title_font * 1.25 + 6)
        row_px = legend_font + 12
        top_margin = int(title_px + rows * row_px + 18)
        height = height + max(0, rows - 1) * row_px
        fig.update_layout(
            height=height,
            title=dict(
                text=title,
                font=dict(
                    family=_TITLE_FONT_FAMILY, size=title_font, color=AXIS_COLOR
                ),
                x=0.5,
                xanchor="center",
                yref="container",
                yanchor="top",
                y=1.0 - title_off / height,
                pad=dict(t=0, b=0),
            ),
            legend=dict(
                orientation="h",
                xref="container",
                yref="container",
                yanchor="top",
                y=1.0 - (title_px + 4) / height,
                xanchor="left",
                x=84 / width,
                bgcolor="rgba(255,255,255,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(size=legend_font),
                itemsizing="constant",
                tracegroupgap=6,
            ),
            margin=dict(l=84, r=40, t=top_margin, b=64),
        )
    return fig


def style_bar_traces(
    fig: go.Figure,
    *,
    color_map: Optional[dict] = None,
    pattern_map: Optional[dict] = None,
    pattern_seq: bool = False,
) -> go.Figure:
    """Apply the shared dark outline + CVD hatch to every bar trace already
    on ``fig`` (e.g. one produced by ``px.bar``).

    ``color_map`` recolours traces by ``trace.name`` (sector → kept hue).
    ``pattern_map`` sets a per-name hatch shape; ``pattern_seq`` instead
    cycles :data:`PATTERN_SHAPES` across traces in order. The hatch
    solidity/size/fg is set uniformly so a series stays separable in
    greyscale.
    """
    bar_idx = 0
    for tr in fig.data:
        if getattr(tr, "type", None) != "bar":
            continue
        name = getattr(tr, "name", None)
        line = dict(color=BAR_LINE_COLOR, width=BAR_LINE_WIDTH)
        if color_map and name in color_map:
            tr.marker.color = color_map[name]
        tr.marker.line = line
        shape = None
        if pattern_map is not None and name in pattern_map:
            shape = pattern_map[name]
        elif pattern_seq:
            shape = PATTERN_SHAPES[bar_idx % len(PATTERN_SHAPES)]
        if shape:
            tr.marker.pattern = dict(
                shape=shape, solidity=0.30, size=7,
                fgcolor="white", fgopacity=0.55,
            )
        bar_idx += 1
    return fig


def empty_fig(message: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(family=_FONT_FAMILY, size=ANNOTATION_FONT_SIZE, color=MUTED_COLOR),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_theme(fig, title=title)


def error_bar(array, arrayminus=None, *, axis: str = "x") -> dict:
    """Uniform error-bar dict matching the scare bar styling."""
    d = dict(
        type="data", array=list(array),
        thickness=1.2, width=4, color=MUTED_COLOR,
    )
    if arrayminus is not None:
        d.update(symmetric=False, arrayminus=list(arrayminus))
    return d
