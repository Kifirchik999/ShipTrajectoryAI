from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.anomaly.detector import detect_rule_based_anomalies
from src.data.loader import (
    standardize_column_names,
    validate_required_columns,
)
from src.features.build_features import build_navigation_features
from src.preprocessing.cleaning import clean_ais_data


DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ais_sample.csv"
)

ISOLATION_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "isolation_forest_results.csv"
)


def prepare_rule_based_data(data):
    data = standardize_column_names(data)
    validate_required_columns(data)
    data = clean_ais_data(data)
    data = build_navigation_features(data)
    data = detect_rule_based_anomalies(data)

    return data


def create_anomaly_map(
    data,
    anomaly_column,
    title,
):
    if data.empty:
        return go.Figure()

    plot_data = data.copy()

    plot_data["Статус"] = plot_data[
        anomaly_column
    ].map(
        {
            False: "Нормальная точка",
            True: "Аномалия",
        }
    )

    hover_columns = [
        column
        for column in [
            "timestamp",
            "mmsi",
            "trajectory_id",
            "sog",
            "cog",
            "course_change",
            "speed_change",
            "calculated_speed_kmh",
            "anomaly_score",
            "anomaly_reason",
        ]
        if column in plot_data.columns
    ]

    figure = px.scatter_map(
        plot_data,
        lat="latitude",
        lon="longitude",
        color="Статус",
        hover_data=hover_columns,
        zoom=5,
        height=650,
        title=title,
    )

    figure.add_trace(
        go.Scattermap(
            lat=plot_data["latitude"],
            lon=plot_data["longitude"],
            mode="lines",
            name="Траектория",
            line={
                "width": 2,
            },
        )
    )

    figure.update_layout(
        map_style="open-street-map",
        margin={
            "r": 0,
            "t": 45,
            "l": 0,
            "b": 0,
        },
    )

    return figure


def select_trajectory(data):
    if "trajectory_id" not in data.columns:
        return data

    trajectory_ids = sorted(
        data["trajectory_id"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_trajectory = st.selectbox(
        "Выберите траекторию",
        options=trajectory_ids,
    )

    return data[
        data["trajectory_id"]
        == selected_trajectory
    ].copy()


st.set_page_config(
    page_title="Обнаружение аномалий",
    page_icon="⚠️",
    layout="wide",
)

st.title("⚠️ Обнаружение навигационных аномалий")

st.caption(
    "Сравнение интерпретируемых правил и алгоритма "
    "Isolation Forest"
)

with st.sidebar:
    st.header("Источник данных")

    uploaded_file = st.file_uploader(
        "Загрузите AIS CSV",
        type=["csv"],
    )

    method = st.radio(
        "Метод обнаружения",
        options=[
            "Rule-based",
            "Isolation Forest",
        ],
    )


if uploaded_file is not None:
    raw_data = pd.read_csv(
        uploaded_file,
        sep=None,
        engine="python",
    )
else:
    if not DEFAULT_DATA_PATH.exists():
        st.error(
            "Файл data/processed/ais_sample.csv не найден"
        )
        st.stop()

    raw_data = pd.read_csv(
        DEFAULT_DATA_PATH,
        low_memory=False,
    )


if method == "Rule-based":
    try:
        result_data = prepare_rule_based_data(
            raw_data
        )

    except Exception as error:
        st.error(f"Ошибка обработки данных: {error}")
        st.exception(error)
        st.stop()

    anomaly_column = "is_anomaly"
    method_title = "Rule-based detector"

else:
    if uploaded_file is not None:
        st.warning(
            "Isolation Forest сейчас использует заранее "
            "рассчитанные результаты для ais_sample.csv."
        )

    if not ISOLATION_RESULTS_PATH.exists():
        st.error(
            "Файл isolation_forest_results.csv не найден. "
            "Сначала запустите train_isolation_forest.py"
        )
        st.stop()

    result_data = pd.read_csv(
        ISOLATION_RESULTS_PATH,
        parse_dates=["timestamp"],
        low_memory=False,
    )

    anomaly_column = "isolation_anomaly"
    method_title = "Isolation Forest"


result_data[anomaly_column] = (
    result_data[anomaly_column]
    .astype(str)
    .str.lower()
    .map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )
    .fillna(False)
)


total_points = len(result_data)
anomaly_count = int(
    result_data[anomaly_column].sum()
)
anomaly_share = (
    anomaly_count / total_points * 100
    if total_points
    else 0
)

trajectory_count = (
    result_data["trajectory_id"].nunique()
    if "trajectory_id" in result_data.columns
    else 1
)


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Количество точек",
    f"{total_points:,}".replace(",", " "),
)

metric_2.metric(
    "Количество траекторий",
    trajectory_count,
)

metric_3.metric(
    "Обнаружено аномалий",
    anomaly_count,
)

metric_4.metric(
    "Доля аномалий",
    f"{anomaly_share:.2f}%",
)


selected_data = select_trajectory(
    result_data
)

selected_anomaly_count = int(
    selected_data[anomaly_column].sum()
)


st.info(
    f"Для выбранной траектории обнаружено "
    f"{selected_anomaly_count} аномальных точек."
)


tab_map, tab_table, tab_analysis = st.tabs(
    [
        "Карта",
        "Таблица аномалий",
        "Анализ признаков",
    ]
)


with tab_map:
    figure = create_anomaly_map(
        selected_data,
        anomaly_column,
        method_title,
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


with tab_table:
    anomalies = selected_data[
        selected_data[anomaly_column]
    ].copy()

    table_columns = [
        column
        for column in [
            "timestamp",
            "trajectory_id",
            "mmsi",
            "latitude",
            "longitude",
            "sog",
            "cog",
            "speed_change",
            "course_change",
            "calculated_speed_kmh",
            "anomaly_score",
            "anomaly_reason",
        ]
        if column in anomalies.columns
    ]

    if anomalies.empty:
        st.success(
            "В выбранной траектории аномалии не найдены"
        )
    else:
        if (
            method == "Isolation Forest"
            and "anomaly_score" in anomalies.columns
        ):
            anomalies = anomalies.sort_values(
                "anomaly_score",
                ascending=False,
            )

        st.dataframe(
            anomalies[table_columns],
            width="stretch",
            hide_index=True,
        )


with tab_analysis:
    feature_columns = [
        column
        for column in [
            "sog",
            "speed_change",
            "course_change",
            "distance_from_previous_km",
            "calculated_speed_kmh",
            "time_delta_seconds",
        ]
        if column in result_data.columns
    ]

    if not feature_columns:
        st.info(
            "Недостаточно признаков для анализа"
        )
    else:
        comparison = (
            result_data.groupby(
                anomaly_column
            )[feature_columns]
            .mean()
            .T
            .reset_index()
        )

        comparison.columns = [
            "Признак",
            "Нормальные точки",
            "Аномальные точки",
        ]

        st.subheader(
            "Средние значения признаков"
        )

        st.dataframe(
            comparison.round(4),
            width="stretch",
            hide_index=True,
        )

        long_comparison = comparison.melt(
            id_vars="Признак",
            value_vars=[
                "Нормальные точки",
                "Аномальные точки",
            ],
            var_name="Тип точки",
            value_name="Среднее значение",
        )

        feature_figure = px.bar(
            long_comparison,
            x="Признак",
            y="Среднее значение",
            color="Тип точки",
            barmode="group",
        )

        st.plotly_chart(
            feature_figure,
            width="stretch",
        )


st.subheader("Интерпретация метода")

if method == "Rule-based":
    st.success(
        "Rule-based detector использует заранее заданные "
        "понятные правила: резкое изменение курса, высокая "
        "скорость, скачок координат и временной разрыв."
    )

else:
    st.success(
        "Isolation Forest ищет редкие комбинации "
        "навигационных признаков. В текущем эксперименте "
        "аномальные точки характеризуются резкими поворотами "
        "и нетипичными изменениями скорости."
    )



