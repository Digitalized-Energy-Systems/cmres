from pathlib import Path

import pandas
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

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
NETWORK_COLOR_MAP_NUM = {"1": "#d32f2f", "2": "#388e3c", "0": "#ffa000"}
AR_COLOR_MAP = {
    0.1: "rgb(65,182,196)",
    0.5: "rgb(34,94,168)",
    0.9: "rgb(8,29,88)",
}

START_ALL_IN_ONE = '<h1>{}</h1><div style="display: flex;align-items: center;flex-direction: row;flex-wrap: wrap;justify-content: space-around;">'
END_ALL_IN_ONE = "</div>"


def write_all_in_one(
    figures, scenario_name, out_path, out_filename, write_single_files=True
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
        for fig in figures:
            fig.write_image(
                path_single_files
                / (
                    fig.layout.title.text
                    + "-"
                    + fig.layout.xaxis.title.text
                    + "-"
                    + fig.layout.yaxis.title.text
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
        range_x=(0, range_x),
        range_y=(0, range_y),
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
        legend={"title": legend_text, "y": 0.2, "x": 0.8},
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
