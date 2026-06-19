import re
import unicodedata
from pathlib import Path

import pandas
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import networkx as nx
import monee

import numpy as np

pio.kaleido.scope.mathjax = None

# ─────────────────────────────────────────────────────────────────────────────
# CMRES unified plot style
#
# One canonical Plotly template plus shared palettes. Every eval/figure
# script in this repo should pull style from here so the HTML/PDF
# outputs look like they belong to the same paper:
#
#     fig.update_layout(template="cmres")           # base template
#     marker_color=PALETTE_QUAL[i % len(PALETTE_QUAL)]
#     colorscale=PALETTE_SEQUENTIAL                  # heatmaps
#     colorscale=PALETTE_DIVERGING                   # signed deltas
#     fig.update_layout(**LEGEND_RIGHT)              # outside-right legend
#
# Carrier-coloured plots keep using ``NETWORK_COLOR_MAP``.
# ─────────────────────────────────────────────────────────────────────────────

_FONT_FAMILY = "Inter, Helvetica Neue, Helvetica, Arial, sans-serif"
_FONT_COLOR = "#1f2933"
_AXIS_LINE = "#cbd2d7"
_AXIS_GRID = "#e6e8eb"
_LEGEND_BG = "rgba(255,255,255,0.85)"
_LEGEND_BORDER = "#cbd2d7"

# Qualitative palette — Tableau-10 inspired, ColorBrewer-vetted hues. Anchored
# so the first three colours match common carrier conventions (blue ~ data,
# orange ~ electricity, green ~ gas); for actual carrier plots use
# ``NETWORK_COLOR_MAP`` directly.
PALETTE_QUAL = [
    "#4c78a8",  # blue
    "#f58518",  # orange
    "#54a24b",  # green
    "#e45756",  # red
    "#72b7b2",  # teal
    "#eeca3b",  # yellow
    "#b279a2",  # purple
    "#ff9da6",  # pink
    "#9d755d",  # brown
    "#bab0ac",  # gray
]
PALETTE_SEQUENTIAL = px.colors.sequential.Viridis
PALETTE_DIVERGING = px.colors.diverging.RdBu

_CMRES_AXIS = dict(
    linecolor=_AXIS_LINE,
    linewidth=1,
    gridcolor=_AXIS_GRID,
    gridwidth=1,
    zerolinecolor=_AXIS_GRID,
    zerolinewidth=1,
    ticks="outside",
    tickcolor=_AXIS_LINE,
    ticklen=4,
    tickfont=dict(size=12, family=_FONT_FAMILY, color=_FONT_COLOR),
    title=dict(font=dict(size=13, family=_FONT_FAMILY, color=_FONT_COLOR)),
    mirror=False,
    automargin=True,
)

pio.templates["cmres"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=_FONT_FAMILY, size=14, color=_FONT_COLOR),
        title=dict(
            font=dict(family=_FONT_FAMILY, size=16, color=_FONT_COLOR),
            x=0.5,
            xanchor="center",
        ),
        colorway=PALETTE_QUAL,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=72, r=32, t=70, b=64),
        xaxis=_CMRES_AXIS,
        yaxis=_CMRES_AXIS,
        legend=dict(
            bgcolor=_LEGEND_BG,
            bordercolor=_LEGEND_BORDER,
            borderwidth=0,
            font=dict(size=12, family=_FONT_FAMILY, color=_FONT_COLOR),
            title=dict(font=dict(size=12, family=_FONT_FAMILY, color=_FONT_COLOR)),
        ),
        hoverlabel=dict(
            font=dict(family=_FONT_FAMILY, size=12),
            bgcolor="white",
            bordercolor=_AXIS_LINE,
        ),
        colorscale=dict(
            sequential=PALETTE_SEQUENTIAL,
            sequentialminus=list(reversed(PALETTE_SEQUENTIAL)),
            diverging=PALETTE_DIVERGING,
        ),
    )
)

# Backwards-compatible aliases so any unsuspecting "publish*" reference
# still resolves to the unified style instead of overriding font size.
pio.templates["publish"] = pio.templates["cmres"]
pio.templates["publish1"] = pio.templates["cmres"]
pio.templates["publish2"] = pio.templates["cmres"]
pio.templates["publish3"] = pio.templates["cmres"]

#: Default Plotly template name for every cmres figure.
CMRES_TEMPLATE = "cmres"

#: Drop-in ``fig.update_layout(**LEGEND_RIGHT)`` to anchor legends consistently
#: outside-right with a transparent background — preferred for medium/wide
#: panels.
LEGEND_RIGHT = dict(
    legend=dict(
        x=1.02,
        xanchor="left",
        y=1.0,
        yanchor="top",
        bgcolor=_LEGEND_BG,
        bordercolor=_LEGEND_BORDER,
        borderwidth=0,
        font=dict(size=12, family=_FONT_FAMILY, color=_FONT_COLOR),
        title=dict(font=dict(size=12, family=_FONT_FAMILY, color=_FONT_COLOR)),
    ),
)

#: Use for grouped categories where horizontal placement reads better than
#: a stacked column legend. Mirrors LEGEND_RIGHT typography.
LEGEND_BOTTOM = dict(
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.20,
        yanchor="top",
        bgcolor=_LEGEND_BG,
        bordercolor=_LEGEND_BORDER,
        borderwidth=0,
        font=dict(size=12, family=_FONT_FAMILY, color=_FONT_COLOR),
        title=dict(font=dict(size=12, family=_FONT_FAMILY, color=_FONT_COLOR)),
    ),
)


def apply_cmres_style(
    fig: "go.Figure",
    legend: str = "right",
    **overrides,
) -> "go.Figure":
    """Apply the canonical cmres template + legend placement to *fig*.

    Args:
        fig: Plotly figure to restyle in place.
        legend: ``"right"`` (default), ``"bottom"`` or ``"none"`` (hide).
        **overrides: forwarded straight to ``fig.update_layout`` so callers can
            tweak titles/axes without re-specifying the template.

    Returns:
        The same figure (so calls can be chained).
    """
    fig.update_layout(template=CMRES_TEMPLATE)
    if legend == "right":
        fig.update_layout(**LEGEND_RIGHT)
    elif legend == "bottom":
        fig.update_layout(**LEGEND_BOTTOM)
    elif legend == "none":
        fig.update_layout(showlegend=False)
    if overrides:
        fig.update_layout(**overrides)
    return fig


YlGnBuDark = [
    "rgb(199,233,180)",
    "rgb(127,205,187)",
    "rgb(65,182,196)",
    "rgb(29,145,192)",
    "rgb(34,94,168)",
    "rgb(37,52,148)",
    "rgb(8,29,88)",
]

COLOR_SCALE_TIME = px.colors.sample_colorscale(px.colors.sequential.Plasma_r, 96)
COLOR_SCALE_AR = px.colors.sample_colorscale(px.colors.sequential.Plasma_r, 100)
COLOR_SCALE_AR_10 = px.colors.sample_colorscale(px.colors.sequential.Plasma_r, 10)
COLOR_SCALE_YB_3 = px.colors.sample_colorscale(YlGnBuDark, 3)

CP_TYPE_COLOR_MAP = {"p2h": "#5e35b1", "p2g": "#00897b"}
NETWORK_COLOR_MAP = {"heat": "#d32f2f", "gas": "#388e3c", "electricity": "#ffa000"}
NETWORK_PATTERN_MAP = {"heat": ".", "gas": "\\", "electricity": "+"}
NETWORK_COLOR_MAP_NUM = {"1": "#d32f2f", "2": "#388e3c", "0": "#ffa000"}
AR_COLOR_MAP = {
    0.1: "rgb(65,182,196)",
    0.5: "rgb(34,94,168)",
    0.9: "rgb(8,29,88)",
}

START_ALL_IN_ONE = '<h1>{}</h1><div style="display: flex;align-items: center;flex-direction: row;flex-wrap: wrap;justify-content: space-around;">'
END_ALL_IN_ONE = "</div>"


def get_title(fig, index, titles):
    if hasattr(fig.layout, "title") and fig.layout.title.text:
        return fig.layout.title.text
    return titles[index]


# Title fragments after the first long-dash (em-dash, en-dash, " — ", " - ")
# are typically run-time-computed annotation strings (Spearman ρ, p-values,
# bootstrap CIs). Keeping them in filenames makes the same figure produce a
# *different* filename on every run, which in turn breaks reproducibility for
# the paper draft. Strip them before slugifying.
_STATS_SUFFIX_RE = re.compile(r"\s+(?:—|–|-)\s+.*$")
# Anything that's not a-z, 0-9, "_" or "-" gets replaced with "_". Greek/math
# characters end up stripped — by design, slug filenames stay ASCII.
_NON_SLUG_CHAR_RE = re.compile(r"[^a-z0-9_-]+")
_MULTI_USCORE_RE = re.compile(r"_+")
_MULTI_DASH_RE = re.compile(r"-+")
_MAX_SLUG_LEN = 60


def slugify(s: str) -> str:
    """Aggressive ASCII-only slug for use in filenames.

    Preserves the legacy behaviour of stripping ``/<>`` so any code that
    still calls slugify directly does not silently regress; everything else
    is normalised to ``[a-z0-9_-]``.
    """
    if s is None:
        return ""
    # NFKD decomposes accented characters into ASCII + combining marks; the
    # subsequent encode("ascii", "ignore") drops the non-ASCII parts.
    s = unicodedata.normalize("NFKD", str(s))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = _NON_SLUG_CHAR_RE.sub("_", s)
    s = _MULTI_USCORE_RE.sub("_", s)
    s = _MULTI_DASH_RE.sub("-", s)
    return s.strip("_-")


def _figure_slug(fig, index: int, titles, slugs):
    """Pick a short, filesystem-safe identifier for one figure.

    Resolution order:
      1. ``slugs[index]`` if the caller supplied a slugs list with this entry
         set to a non-empty string — used as-is after a defensive slugify.
      2. The figure's layout title or ``titles[index]``, with the run-time
         stats-line suffix (after " — ", " - ", or " - ") stripped, then
         aggressively slugified and truncated to ``_MAX_SLUG_LEN`` chars.
      3. ``"figure"`` as the last-resort default.

    The numeric index prefix is added by the caller so collisions across
    differently-titled figures in the same HTML are impossible.
    """
    if slugs is not None and index < len(slugs) and slugs[index]:
        explicit = slugify(slugs[index])
        if explicit:
            return explicit[:_MAX_SLUG_LEN]

    raw_title = None
    if fig is not None and hasattr(fig, "layout") and getattr(fig.layout, "title", None):
        raw_title = getattr(fig.layout.title, "text", None)
    if not raw_title and titles is not None and index < len(titles):
        raw_title = titles[index]
    if not raw_title:
        return "figure"
    cleaned = _STATS_SUFFIX_RE.sub("", str(raw_title))
    slug = slugify(cleaned)
    if not slug:
        return "figure"
    # Cut at a word boundary if possible so we don't end mid-token.
    if len(slug) > _MAX_SLUG_LEN:
        slug = slug[:_MAX_SLUG_LEN]
        last_us = slug.rfind("_")
        if last_us > _MAX_SLUG_LEN // 2:
            slug = slug[:last_us]
    return slug or "figure"


def write_all_in_one(
    figures,
    scenario_name,
    out_path,
    out_filename,
    write_single_files=True,
    titles=None,
    slugs=None,
):
    """Write a combined HTML and (optionally) one PDF per figure.

    Single-file PDFs are named ``{NN}_{slug}.pdf`` where ``NN`` is the
    figure's zero-padded index in ``figures`` and ``slug`` is derived from
    ``slugs[i]`` (if provided) or the figure title with the run-time
    stats-line suffix stripped. The index prefix preserves the document
    order and guarantees uniqueness, so the previous ``-{xaxis}-{yaxis}``
    disambiguator is no longer needed.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / out_filename).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path / out_filename, "w", encoding="utf-8") as file:
        file.write(START_ALL_IN_ONE.format(scenario_name))
        file.write(figures[0].to_html(include_plotlyjs="cdn"))
        for fig in figures[1:]:
            file.write(fig.to_html(full_html=False, include_plotlyjs=False))
        file.write(END_ALL_IN_ONE)

    # workaround loading box error
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image("random_figure.pdf", format="pdf")
    Path("random_figure.pdf").unlink()

    if write_single_files:
        path_single_files = (out_path / out_filename).parent / "single"
        path_single_files.mkdir(parents=True, exist_ok=True)
        # Width of the index prefix scales with figure count so order is
        # preserved by lexicographic sort even for 100+-figure outputs.
        idx_width = max(2, len(str(max(0, len(figures) - 1))))
        for i, fig in enumerate(figures):
            slug = _figure_slug(fig, i, titles, slugs)
            filename = f"{i:0{idx_width}d}_{slug}.pdf"
            fig.write_image(path_single_files / filename)


def create_group_histogram(
    df,
    x_label,
    y_label,
    color,
    height=400,
    width=600,
    template=CMRES_TEMPLATE,
    range_x=None,
    range_y=None,
    title=None,
    legend_text=None,
    xaxis_title=None,
    yaxis_title=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
):
    fig = px.histogram(
        df,
        x=x_label,
        y=y_label,
        color=color,
        title=title,
        template=template,
        barmode="group",
        color_discrete_sequence=color_discrete_sequence,
        color_discrete_map=color_discrete_map,
    )
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 20, "b": 30, "r": 10, "t": 30},
        legend={"title": legend_text},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_tickangle=-45,
    )
    return fig


def create_bar(
    df,
    x_label,
    y_label,
    color=None,
    legend_text=None,
    height=400,
    width=600,
    template=CMRES_TEMPLATE,
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    pattern_shape_map=None,
    marker_color=None,
    barmode=None,
    showlegend=True,
):
    fig = px.bar(
        df,
        x=x_label,
        y=y_label,
        color=color,
        title=title,
        template=template,
        color_discrete_sequence=color_discrete_sequence,
        color_discrete_map=color_discrete_map,
        pattern_shape_map=pattern_shape_map,
        barmode=barmode,
    )
    if marker_color is not None:
        fig.update_traces(marker_color=marker_color)
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 20, "b": 30, "r": 10, "t": 30},
        legend={"title": legend_text},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_tickangle=-45,
        showlegend=showlegend,
    )
    return fig


def create_multi_bar(
    name_hist_list,
    x=None,
    height=400,
    width=600,
    template=CMRES_TEMPLATE,
    title=None,
    legend_text=None,
    xaxis_title=None,
    yaxis_title=None,
    offsetgroup=0,
):
    fig = go.Figure()
    for name, y in name_hist_list:
        fig.add_trace(go.Bar(x=x, y=y, name=name, offsetgroup=offsetgroup))
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 20, "b": 30, "r": 10, "t": 30},
        legend={"title": legend_text},
        template=template,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig


def create_time_series(
    dff,
    index,
    title=None,
    height=400,
    width=600,
    template=CMRES_TEMPLATE,
    legend_text=None,
    xaxis_title=None,
    yaxis_title=None,
):
    x, y, ax = dff
    if len(x) == 0:
        fig = px.line(x=[0, 1], y=[0, 1])
        return fig

    if isinstance(y[index], dict):
        data_frame_dict = y[index]
        fig = px.line(
            pandas.DataFrame(data_frame_dict),
            template=template,
            title=title,
        )
    else:
        fig = px.scatter(
            pandas.DataFrame({"unit": y[index], "time": x[index]}),
            x="time",
            y="unit",
            template=template,
            title=title,
        )

    fig.update_traces(mode="lines+markers")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(type="linear")
    if title is None:
        fig.add_annotation(
            x=0,
            y=0.85,
            xanchor="left",
            yanchor="bottom",
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            text=ax[index],
        )
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 20, "b": 30, "r": 10, "t": 30},
        legend={"title": legend_text},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig


def create_line_with_df(
    df,
    x_label,
    y_label,
    color_label,
    color_discrete_sequence=None,
    title=None,
    height=400,
    width=600,
    template=CMRES_TEMPLATE,
    legend_text=None,
    xaxis_title=None,
    yaxis_title=None,
    line_dash_sequence=None,
    line_dash=None,
    line_width=None,
):
    fig = px.line(
        df,
        x=x_label,
        y=y_label,
        color=color_label,
        color_discrete_sequence=color_discrete_sequence,
        template=template,
        title=title,
        line_dash_sequence=line_dash_sequence,
        line_dash=line_dash,
    )
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 30, "b": 40, "r": 20, "t": 40},
        legend={"title": legend_text},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    fig.update_traces(line=dict(width=line_width))
    return fig


def create_scatter_with_df(
    df,
    x_label,
    y_label,
    color_label,
    color_discrete_sequence=None,
    color_discrete_map=None,
    title=None,
    height=400,
    width=600,
    template=CMRES_TEMPLATE,
    legend_text=None,
    xaxis_title=None,
    yaxis_title=None,
    trendline=None,
    trendline_options=None,
    symbol_seq=["circle-open", "x", "diamond-wide-open"],
    symbol=-1,
    log_x=False,
    log_y=False,
    color_continous_scale=None,
    mode=None,
):
    if symbol == -1:
        symbol = color_label
    fig = px.scatter(
        df,
        x=x_label,
        y=y_label,
        color=color_label,
        color_discrete_sequence=color_discrete_sequence,
        color_discrete_map=color_discrete_map,
        color_continuous_scale=color_continous_scale,
        template=template,
        title=title,
        trendline=trendline,
        trendline_options=trendline_options,
        symbol=symbol,
        symbol_sequence=symbol_seq,
        log_x=log_x,
        log_y=log_y,
    )
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 30, "b": 40, "r": 20, "t": 40},
        legend={"title": legend_text, "y": 0, "x": 1},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    if color_discrete_map is None:
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=legend_text,
            ),
        )
        fig.layout.coloraxis.colorbar.thickness = 15
        fig.layout.coloraxis.colorbar.xanchor = "left"
        fig.layout.coloraxis.colorbar.title.side = "right"
        fig.layout.coloraxis.colorbar.outlinewidth = 2
        fig.layout.coloraxis.colorbar.outlinecolor = "#888"
    if mode is not None:
        fig.data[0].mode = mode
    return fig


GRID_NAME_TO_SHIFT_X = {
    "power": 0,
    "heat": 0.0003,
    "water": 0.0003,
    "gas": 0.0006,
    "None": 0.0003,
    None: 0.0003,
}
GRID_NAME_TO_SHIFT_Y = {
    "power": 0,
    "heat": 0.0003,
    "water": 0.0003,
    "gas": 0.0006,
    "None": -0.0003,
    None: -0.0003,
}


def create_networkx_plot(
    network: monee.Network,
    df,
    color_name,
    color_legend_text=None,
    title=None,
    template=CMRES_TEMPLATE,
    without_nodes=False,
):
    graph: nx.Graph = network._network_internal
    # pre-compute spring layout for nodes that have no position
    fallback_pos = nx.spring_layout(graph, seed=42)

    # ── Pre-built lookups (replace per-edge / per-node O(N_df) scans) ──────
    # 1) id -> color: one pass over df instead of df.loc per element.
    if color_name in df.columns and "id" in df.columns:
        color_lookup = (
            df[["id", color_name]]
            .dropna(subset=["id"])
            .drop_duplicates(subset="id", keep="first")
            .set_index("id")[color_name]
            .to_dict()
        )
    else:
        color_lookup = {}
    # 2) graph node id -> monee node: avoid network.node_by_id() per edge.
    m_node_cache = {nid: network.node_by_id(nid) for nid in graph.nodes}

    pos = {}
    x_edges = []
    y_edges = []
    color_edges = []
    for from_node, to_node, uid in graph.edges:
        from_m_node = m_node_cache[from_node]
        to_m_node = m_node_cache[to_node]
        from_grid = from_m_node.grid[0] if isinstance(from_m_node.grid, list) else from_m_node.grid
        to_grid = to_m_node.grid[0] if isinstance(to_m_node.grid, list) else to_m_node.grid
        add_to_from_x = GRID_NAME_TO_SHIFT_X[from_grid.name]
        add_to_from_y = GRID_NAME_TO_SHIFT_Y[from_grid.name]
        add_to_to_x = GRID_NAME_TO_SHIFT_X[to_grid.name]
        add_to_to_y = GRID_NAME_TO_SHIFT_Y[to_grid.name]
        from_pos = from_m_node.position if from_m_node.position is not None else fallback_pos[from_node]
        to_pos = to_m_node.position if to_m_node.position is not None else fallback_pos[to_node]
        x0, y0 = (
            from_pos[0] + add_to_from_x,
            from_pos[1] + add_to_from_y,
        )
        x1, y1 = (
            to_pos[0] + add_to_to_x,
            to_pos[1] + add_to_to_y,
        )
        pos[from_node] = (x0, y0)
        pos[to_node] = (x1, y1)
        color_data = color_lookup.get(
            f"branch:({from_node}, {to_node}, {uid})", 0
        )

        x_edges.append([x0, x1, None])
        y_edges.append([y0, y1, None])
        color_edges.append(color_data)
    node_x_power = []
    node_y_power = []
    node_color_power = []
    node_text_power = []
    node_x_heat = []
    node_y_heat = []
    node_color_heat = []
    node_text_heat = []
    node_x_gas = []
    node_y_gas = []
    node_color_gas = []
    node_text_gas = []
    node_cp_x = []
    node_cp_y = []
    node_color_cp = []
    node_text_cp = []
    for node in graph.nodes:
        node_id = f"node:{node}"
        x, y = pos.get(node, fallback_pos[node])
        node_data = graph.nodes[node]
        int_node = node_data["internal_node"]
        color_data = color_lookup.get(node_id, 0)
        node_text = (
            str(type(int_node.grid).__name__)
            + " - "
            + str(type(int_node.model).__name__)
            + " - "
            + str(color_data)
        )
        if not int_node.independent:
            node_cp_x.append(x)
            node_cp_y.append(y)
            node_color_cp.append(color_data)
            node_text_cp.append(node_text)
        elif "Water" in str(type(int_node.grid)):
            node_x_heat.append(x)
            node_y_heat.append(y)
            node_color_heat.append(color_data)
            node_text_heat.append(node_text)
        elif "Gas" in str(type(int_node.grid)):
            node_x_gas.append(x)
            node_y_gas.append(y)
            node_color_gas.append(color_data)
            node_text_gas.append(node_text)
        elif "Power" in str(type(int_node.grid)):
            node_x_power.append(x)
            node_y_power.append(y)
            node_color_power.append(color_data)
            node_text_power.append(node_text)

    max_color_val = max(
        color_edges
        if without_nodes
        else node_color_gas
        + node_color_cp
        + node_color_heat
        + node_color_power
        + color_edges
    )
    # Group edges into a small number of color bins so plotly only renders
    # ~N_BINS Scatter traces instead of one per edge. With 300+ edges, the
    # per-trace overhead dominates render time; binning shrinks that 20×.
    # The binning is visually equivalent to the original gradient at
    # screen resolution.
    N_BINS = 20
    edge_traces = []
    if max_color_val == 0:
        # Single uniform black trace, all edges flattened.
        all_x: list = []
        all_y: list = []
        for ex, ey in zip(x_edges, y_edges):
            all_x.extend(ex)
            all_y.extend(ey)
        edge_traces.append(
            go.Scatter(
                x=all_x, y=all_y,
                line=dict(width=3, color="rgb(0,0,0)"),
                hoverinfo="skip",
                mode="lines",
                showlegend=False,
            )
        )
    else:
        # Pre-sample the colorscale once at N_BINS evenly spaced points.
        bin_samples = [t / max(N_BINS - 1, 1) for t in range(N_BINS)]
        bin_colors = px.colors.sample_colorscale(
            px.colors.sequential.Sunsetdark, bin_samples
        )
        # Bucket edges by which sampled color they belong to.
        bin_x: list = [[] for _ in range(N_BINS)]
        bin_y: list = [[] for _ in range(N_BINS)]
        for ex, ey, ec in zip(x_edges, y_edges, color_edges):
            t = min(1.0, max(0.0, ec / max_color_val))
            bi = min(N_BINS - 1, int(round(t * (N_BINS - 1))))
            bin_x[bi].extend(ex)
            bin_y[bi].extend(ey)
        for bi in range(N_BINS):
            if not bin_x[bi]:
                continue
            edge_traces.append(
                go.Scatter(
                    x=bin_x[bi], y=bin_y[bi],
                    line=dict(width=3, color=bin_colors[bi]),
                    hoverinfo="skip",
                    mode="lines",
                    showlegend=False,
                )
            )

    # cp
    node_trace_cp = go.Scatter(
        x=node_cp_x,
        y=node_cp_y,
        mode="markers",
        hoverinfo="text",
        text=node_text_cp,
        marker=dict(
            color=node_color_cp,
            symbol="diamond",
            size=9,
            coloraxis="coloraxis",
            line=dict(width=1, color="#d3d3d3"),
        ),
    )

    # heat
    node_trace_heat = go.Scatter(
        x=node_x_heat,
        y=node_y_heat,
        mode="markers",
        hoverinfo="text",
        text=node_text_heat,
        marker=dict(
            color=node_color_heat,
            symbol="pentagon",
            size=9,
            coloraxis="coloraxis",
            line=dict(width=1, color="#d3d3d3"),
        ),
    )
    # power
    node_trace_power = go.Scatter(
        x=node_x_power,
        y=node_y_power,
        mode="markers",
        hoverinfo="text",
        text=node_text_power,
        marker=dict(
            color=node_color_power,
            symbol="square",
            size=9,
            coloraxis="coloraxis",
            line=dict(width=1, color="#d3d3d3"),
        ),
    )
    # gas
    node_trace_gas = go.Scatter(
        x=node_x_gas,
        y=node_y_gas,
        mode="markers",
        hoverinfo="text",
        text=node_text_gas,
        marker=dict(
            color=node_color_gas,
            symbol="triangle-up",
            size=9,
            coloraxis="coloraxis",
            line=dict(width=1, color="#d3d3d3"),
        ),
    )

    fig = go.Figure(
        data=edge_traces
        + (
            [
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        coloraxis="coloraxis",
                        showscale=True,
                    ),
                    hoverinfo="none",
                )
            ]
            if without_nodes
            else [
                node_trace_heat,
                node_trace_power,
                node_trace_gas,
                node_trace_cp,
            ]
        ),
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template=template,
        ),
    )
    fig.update_layout(
        height=400,
        width=600,
        margin={"l": 20, "b": 30, "r": 10, "t": 30},
        xaxis_title="",
        legend={"title": color_legend_text},
        yaxis_title="",
        title=title,
        coloraxis_colorbar=dict(
            title=color_legend_text,
        ),
    )
    fig.layout.coloraxis.showscale = True
    fig.layout.coloraxis.colorscale = "Sunsetdark"
    fig.layout.coloraxis.reversescale = False
    fig.layout.coloraxis.colorbar.thickness = 15
    fig.layout.coloraxis.colorbar.xanchor = "left"
    fig.layout.coloraxis.colorbar.title.side = "right"
    fig.layout.coloraxis.colorbar.outlinewidth = 2
    fig.layout.coloraxis.colorbar.outlinecolor = "#888"
    fig.layout.coloraxis.cmin = min(
        node_color_gas
        + node_color_cp
        + node_color_heat
        + node_color_power
        + color_edges
    )
    fig.layout.coloraxis.cmax = max_color_val
    return fig


def create_multilevel_grouped_bar_chart(
    y_array_list,
    color_list,
    name_list,
    group_labels,
    group_size,
    x_axis_labels,
    yaxis_title,
    title=None,
    multi_level_distance=-0.18,
):
    fig = go.Figure()
    common_x = np.array(list(range(len(y_array_list[0])))) + np.array(
        [0.5 * (i // group_size) for i in range(len(y_array_list[0]))]
    )

    for i, color in enumerate(color_list):
        fig.add_bar(
            x=common_x,
            y=y_array_list[i],
            name=name_list[i],
            marker_color=color,
        )
    for i, group_label in enumerate(group_labels):
        fig.add_annotation(
            text=group_label,
            xref="paper",
            yref="paper",
            x=(common_x[i * group_size] + 2.5) / (max(common_x)),
            y=multi_level_distance,
            showarrow=False,
            font_size=20,
        )

    # Layout
    fig.update_layout(
        barmode="stack",
        showlegend=True,
        template=CMRES_TEMPLATE,
        height=800,
        width=1600,
        legend=dict(
            title="",
            orientation="h",
            traceorder="normal",
            x=0.46,
            y=1.05,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,1)",
            borderwidth=0,
            font_size=20,
        ),
        title=title,
    )

    fig.update_yaxes(
        showline=True,
        showgrid=False,
        linewidth=0.5,
        linecolor="black",
        title=dict(text=yaxis_title, font=dict(size=24), standoff=40),
        ticks="outside",
        dtick=2,
        ticklen=10,
        tickfont=dict(size=20),
        range=[
            0,
            max(
                [
                    sum([y_array_list[i][j] for i in range(len(y_array_list))])
                    for j in range(len(y_array_list[0]))
                ]
            )
            + 0.5,
        ],
    )

    fig.update_xaxes(
        title="",
        tickvals=common_x,
        ticktext=x_axis_labels,
        ticks="",
        tickfont_size=20,
        linecolor="black",
        linewidth=1,
    )
    return fig
