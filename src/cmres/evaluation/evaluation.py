from pathlib import Path

import pandas
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import networkx.drawing.nx_agraph as nxd
import networkx as nx
import plotly.express as px
import monee

pio.kaleido.scope.mathjax = None

pio.templates["publish"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", size=19),
        titlefont=dict(family="sans-serif", size=19),
    )
)
pio.templates["publish3"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", size=19),
        titlefont=dict(family="sans-serif", size=19),
    )
)
pio.templates["publish2"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", size=13),
        titlefont=dict(family="sans-serif", size=13),
    )
)
pio.templates["publish1"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", size=9),
        titlefont=dict(family="sans-serif", size=9),
    )
)

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

CP_TYPE_COLOR_MAP = {"p2h": "#5e35b1", "p2g": "#00897b", "p2h": "#d81b60"}
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


def slugify(str: str):
    return str.replace("/", "").replace("<", "").replace(">", "")


def write_all_in_one(
    figures, scenario_name, out_path, out_filename, write_single_files=True, titles=None
):
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / out_filename).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path / out_filename, "w") as file:
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
        for i, fig in enumerate(figures):
            fig.write_image(
                path_single_files
                / (
                    get_title(fig, i, titles)
                    + "-"
                    + slugify(fig.layout.xaxis.title.text)
                    + "-"
                    + slugify(fig.layout.yaxis.title.text)
                    + ".pdf"
                )
            )


def create_group_histogram(
    df,
    x_label,
    y_label,
    color,
    height=400,
    width=600,
    template="plotly_white",
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
    template="plotly_white",
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
    template="plotly_white",
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
    template="plotly_white",
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
    template="plotly_white+publish",
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
    template="plotly_white+publish",
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
        fig.layout.coloraxis.colorbar.titleside = "right"
        fig.layout.coloraxis.colorbar.outlinewidth = 2
        fig.layout.coloraxis.colorbar.outlinecolor = "#888"
    if mode is not None:
        fig.data[0].mode = mode
    return fig


GRID_NAME_TO_SHIFT_X = {
    "power": 0,
    "heat": 0.0003,
    "gas": 0.0006,
    "None": 0.0003,
    None: 0.0003,
}
GRID_NAME_TO_SHIFT_Y = {
    "power": 0,
    "heat": 0.0003,
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
    template="plotly_white+publish2",
    without_nodes=False,
):
    graph: nx.Graph = network._network_internal
    # pos = nxd.pygraphviz_layout(graph, prog="neato")
    # pos =
    pos = {}
    x_edges = []
    y_edges = []
    color_edges = []
    for from_node, to_node, uid in graph.edges:
        from_m_node = network.node_by_id(from_node)
        to_m_node = network.node_by_id(to_node)
        add_to_from_x = GRID_NAME_TO_SHIFT_X[from_m_node.grid.name]
        add_to_from_y = GRID_NAME_TO_SHIFT_Y[from_m_node.grid.name]
        add_to_to_x = GRID_NAME_TO_SHIFT_X[to_m_node.grid.name]
        add_to_to_y = GRID_NAME_TO_SHIFT_Y[to_m_node.grid.name]
        x0, y0 = (
            from_m_node.position[0] + add_to_from_x,
            from_m_node.position[1] + add_to_from_y,
        )
        x1, y1 = (
            to_m_node.position[0] + add_to_to_x,
            to_m_node.position[1] + add_to_to_y,
        )
        pos[from_node] = (x0, y0)
        pos[to_node] = (x1, y1)
        color_data = 0
        color_data_list = list(
            df.loc[df["id"] == f"branch:({from_node}, {to_node}, {uid})"][color_name]
        )
        if len(color_data_list) > 0:
            color_data = color_data_list[0]

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
        x, y = pos[node]
        node_data = graph.nodes[node]
        int_node = node_data["internal_node"]
        color_data = 0
        color_data_list = list(df.loc[df["id"] == node_id][color_name])
        if len(color_data_list) > 0:
            color_data = color_data_list[0]
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
    edge_traces = []
    for i in range(len(x_edges)):
        edge_traces.append(
            go.Scatter(
                x=x_edges[i],
                y=y_edges[i],
                line=dict(
                    width=3,
                    color="rgb(0,0,0)"
                    if max(color_edges) == 0
                    else px.colors.sample_colorscale(
                        px.colors.sequential.Sunsetdark,
                        (color_edges[i] / max_color_val) + min(color_edges),
                    )[0],
                ),
                hoverinfo="text",
                mode="lines",
                text=f"{color_edges[i]}",
                marker=dict(
                    coloraxis="coloraxis",
                ),
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
    fig.layout.coloraxis.colorbar.titleside = "right"
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


import numpy as np


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
        template="plotly_white",
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
        title=yaxis_title,
        titlefont=dict(size=24),
        title_standoff=40,
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
