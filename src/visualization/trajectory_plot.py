import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_trajectory_map(
    data: pd.DataFrame,
) -> go.Figure:
    if data.empty:
        return go.Figure()

    plot_data = data.copy()

    color_column = (
        "is_anomaly"
        if "is_anomaly" in plot_data.columns
        else None
    )

    hover_columns = [
        column
        for column in [
            "timestamp",
            "mmsi",
            "sog",
            "cog",
            "risk_score",
            "anomaly_reason",
        ]
        if column in plot_data.columns
    ]

    figure = px.scatter_map(
        plot_data,
        lat="latitude",
        lon="longitude",
        color=color_column,
        hover_data=hover_columns,
        zoom=5,
        height=650,
    )

    figure.add_trace(
        go.Scattermap(
            lat=plot_data["latitude"],
            lon=plot_data["longitude"],
            mode="lines",
            name="Фактическая траектория",
        )
    )

    figure.update_layout(
        map_style="open-street-map",
        margin={
            "r": 0,
            "t": 0,
            "l": 0,
            "b": 0,
        },
        legend_title_text="Аномалия",
    )

    return figure


def create_speed_chart(
    data: pd.DataFrame,
) -> go.Figure:
    if "sog" not in data.columns:
        return go.Figure()

    return px.line(
        data,
        x="timestamp",
        y="sog",
        title="Скорость судна по времени",
        labels={
            "timestamp": "Время",
            "sog": "Скорость, узлы",
        },
    )
