from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.anomaly.detector import detect_rule_based_anomalies
from src.data.loader import (
    standardize_column_names,
    validate_required_columns,
)
from src.features.build_features import build_navigation_features
from src.models.baseline import (
    attach_actual_next_position,
    predict_next_position_constant_velocity,
)
from src.preprocessing.cleaning import clean_ais_data
from src.visualization.trajectory_plot import (
    create_speed_chart,
    create_trajectory_map,
)


def create_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    count = 120

    timestamps = pd.date_range(
        start="2026-07-15 08:00:00",
        periods=count,
        freq="5min",
        tz="UTC",
    )

    latitudes = 54.70 + np.cumsum(
        rng.normal(0.003, 0.0008, count)
    )

    longitudes = 18.65 + np.cumsum(
        rng.normal(0.004, 0.001, count)
    )

    speeds = rng.normal(12, 1.5, count)
    courses = rng.normal(45, 4, count)

    speeds[50] = 42
    courses[80] = 170
    latitudes[95] += 0.18
    longitudes[95] += 0.18

    return pd.DataFrame(
        {
            "mmsi": 273000001,
            "trajectory_id": 0,
            "timestamp": timestamps,
            "latitude": latitudes,
            "longitude": longitudes,
            "sog": speeds,
            "cog": courses,
        }
    )


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    data = standardize_column_names(data)
    validate_required_columns(data)

    data = clean_ais_data(data)
    data = build_navigation_features(data)
    data = predict_next_position_constant_velocity(data)
    data = attach_actual_next_position(data)
    data = detect_rule_based_anomalies(data)

    return data


st.set_page_config(
    page_title="ShipTrajectoryAI",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 ShipTrajectoryAI")

st.caption(
    "Прогнозирование траекторий судов и обнаружение "
    "аномального навигационного поведения"
)

with st.sidebar:
    st.header("Источник данных")

    uploaded_file = st.file_uploader(
        "Загрузите AIS-датасет",
        type=["csv"],
    )

    use_demo_data = st.checkbox(
        "Использовать демонстрационные данные",
        value=uploaded_file is None,
    )

try:
    if uploaded_file is not None:
        raw_data = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python",
        )

        source_name = uploaded_file.name

    elif use_demo_data:
        raw_data = create_demo_data()
        source_name = "Демонстрационный датасет"

    else:
        st.info(
            "Загрузите CSV-файл или включите "
            "демонстрационные данные"
        )
        st.stop()

    processed_data = prepare_data(raw_data)

except Exception as error:
    st.error(f"Ошибка обработки данных: {error}")
    st.exception(error)
    st.stop()


st.success(f"Данные обработаны: {source_name}")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Количество записей",
    f"{len(processed_data):,}".replace(",", " "),
)

vessel_count = (
    processed_data["mmsi"].nunique()
    if "mmsi" in processed_data.columns
    else 1
)

metric_2.metric(
    "Количество судов",
    vessel_count,
)

anomaly_count = int(
    processed_data["is_anomaly"].sum()
)

metric_3.metric(
    "Обнаружено аномалий",
    anomaly_count,
)

anomaly_share = (
    processed_data["is_anomaly"].mean() * 100
)

metric_4.metric(
    "Доля аномалий",
    f"{anomaly_share:.1f}%",
)


if "trajectory_id" in processed_data.columns:
    trajectory_ids = sorted(
        processed_data["trajectory_id"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_trajectory = st.selectbox(
        "Выберите траекторию",
        options=trajectory_ids,
    )

    vessel_data = processed_data[
        processed_data["trajectory_id"]
        == selected_trajectory
    ].copy()

    if "mmsi" in vessel_data.columns and not vessel_data.empty:
        selected_mmsi = vessel_data["mmsi"].iloc[0]
        st.caption(f"MMSI судна: {selected_mmsi}")

elif "mmsi" in processed_data.columns:
    vessel_ids = sorted(
        processed_data["mmsi"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_vessel = st.selectbox(
        "Выберите судно",
        options=vessel_ids,
    )

    vessel_data = processed_data[
        processed_data["mmsi"] == selected_vessel
    ].copy()

else:
    vessel_data = processed_data.copy()


tab_map, tab_charts, tab_anomalies, tab_data = st.tabs(
    [
        "Карта",
        "Графики",
        "Аномалии",
        "Данные",
    ]
)


with tab_map:
    st.subheader("Траектория судна")

    figure = create_trajectory_map(vessel_data)

    st.plotly_chart(
        figure,
        width="stretch",
    )


with tab_charts:
    st.subheader("Навигационные параметры")

    if "sog" in vessel_data.columns:
        st.plotly_chart(
            create_speed_chart(vessel_data),
            width="stretch",
        )

    if "course_change" in vessel_data.columns:
        st.line_chart(
            vessel_data.set_index("timestamp")[
                "course_change"
            ]
        )


with tab_anomalies:
    st.subheader("Обнаруженные аномалии")

    anomalies = vessel_data[
        vessel_data["is_anomaly"]
    ].copy()

    columns = [
        column
        for column in [
            "timestamp",
            "mmsi",
            "trajectory_id",
            "latitude",
            "longitude",
            "sog",
            "cog",
            "course_change",
            "calculated_speed_kmh",
            "risk_score",
            "anomaly_reason",
        ]
        if column in anomalies.columns
    ]

    if anomalies.empty:
        st.success("Аномалии не обнаружены")
    else:
        st.dataframe(
            anomalies[columns],
            width="stretch",
            hide_index=True,
        )


with tab_data:
    st.subheader("Обработанные AIS-данные")

    st.dataframe(
        vessel_data,
        width="stretch",
        hide_index=True,
    )



