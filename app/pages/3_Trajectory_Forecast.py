from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "catboost_predictions.csv"
)


st.set_page_config(
    page_title="Прогноз траектории",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Прогноз траектории судна")

st.caption(
    "Визуальное сравнение прогноза CatBoost, baseline "
    "и фактического будущего положения судна"
)


if not PREDICTIONS_PATH.exists():
    st.error(
        "Файл catboost_predictions.csv не найден. "
        "Сначала запустите src/models/evaluate_catboost.py"
    )
    st.stop()


@st.cache_data
def load_predictions():
    data = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["timestamp"],
        low_memory=False,
    )

    return data.sort_values(
        [
            "horizon_steps",
            "trajectory_id",
            "timestamp",
        ]
    ).reset_index(drop=True)


predictions = load_predictions()


with st.sidebar:
    st.header("Параметры прогноза")

    horizons = sorted(
        predictions["horizon_steps"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    selected_horizon = st.selectbox(
        "Горизонт прогнозирования",
        options=horizons,
        format_func=lambda value: f"{value} AIS-шагов",
    )


horizon_data = predictions[
    predictions["horizon_steps"]
    == selected_horizon
].copy()


trajectory_ids = sorted(
    horizon_data["trajectory_id"]
    .dropna()
    .unique()
    .tolist()
)

selected_trajectory = st.selectbox(
    "Выберите тестовую траекторию",
    options=trajectory_ids,
)


trajectory_data = horizon_data[
    horizon_data["trajectory_id"]
    == selected_trajectory
].copy()

trajectory_data = trajectory_data.sort_values(
    "timestamp"
).reset_index(drop=True)


if trajectory_data.empty:
    st.warning("Для выбранной траектории нет прогнозов")
    st.stop()


selected_index = st.slider(
    "Выберите момент прогнозирования",
    min_value=0,
    max_value=len(trajectory_data) - 1,
    value=min(
        len(trajectory_data) // 2,
        len(trajectory_data) - 1,
    ),
)


selected_row = trajectory_data.iloc[selected_index]

catboost_error_m = (
    selected_row["catboost_error_km"] * 1000
)

baseline_error_m = (
    selected_row["baseline_error_km"] * 1000
)

if baseline_error_m > 0:
    improvement = (
        (
            baseline_error_m
            - catboost_error_m
        )
        / baseline_error_m
        * 100
    )
else:
    improvement = 0


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Горизонт",
    f"{selected_horizon} AIS-шагов",
)

metric_2.metric(
    "Ошибка CatBoost",
    f"{catboost_error_m:.1f} м",
)

metric_3.metric(
    "Ошибка baseline",
    f"{baseline_error_m:.1f} м",
)

metric_4.metric(
    "Изменение ошибки",
    f"{improvement:.1f}%",
)


if "mmsi" in selected_row:
    st.caption(
        f"MMSI: {int(selected_row['mmsi'])} · "
        f"Время прогноза: {selected_row['timestamp']}"
    )


figure = go.Figure()


figure.add_trace(
    go.Scattermap(
        lat=trajectory_data["latitude"],
        lon=trajectory_data["longitude"],
        mode="lines",
        name="Наблюдаемая траектория",
        line={
            "width": 3,
        },
    )
)


figure.add_trace(
    go.Scattermap(
        lat=[
            selected_row["latitude"],
        ],
        lon=[
            selected_row["longitude"],
        ],
        mode="markers",
        name="Текущая позиция",
        marker={
            "size": 14,
        },
        text=[
            "Точка, из которой выполняется прогноз",
        ],
        hoverinfo="text",
    )
)


figure.add_trace(
    go.Scattermap(
        lat=[
            selected_row["predicted_latitude"],
        ],
        lon=[
            selected_row["predicted_longitude"],
        ],
        mode="markers",
        name="Прогноз CatBoost",
        marker={
            "size": 15,
        },
        text=[
            f"Ошибка CatBoost: {catboost_error_m:.1f} м",
        ],
        hoverinfo="text",
    )
)


figure.add_trace(
    go.Scattermap(
        lat=[
            selected_row["actual_latitude"],
        ],
        lon=[
            selected_row["actual_longitude"],
        ],
        mode="markers",
        name="Фактическая будущая позиция",
        marker={
            "size": 15,
        },
        text=[
            "Реальное положение судна "
            f"через {selected_horizon} AIS-шагов",
        ],
        hoverinfo="text",
    )
)


figure.add_trace(
    go.Scattermap(
        lat=[
            selected_row["latitude"],
            selected_row["predicted_latitude"],
        ],
        lon=[
            selected_row["longitude"],
            selected_row["predicted_longitude"],
        ],
        mode="lines",
        name="Направление прогноза",
        line={
            "width": 4,
            "color": "#ff9800",
        },
    )
)


figure.add_trace(
    go.Scattermap(
        lat=[
            selected_row["predicted_latitude"],
            selected_row["actual_latitude"],
        ],
        lon=[
            selected_row["predicted_longitude"],
            selected_row["actual_longitude"],
        ],
        mode="lines",
        name="Ошибка прогноза",
        line={
            "width": 3,
            "color": "#ef5350",
        },
    )
)


center_latitude = float(
    trajectory_data["latitude"].mean()
)

center_longitude = float(
    trajectory_data["longitude"].mean()
)


figure.update_layout(
    map={
        "style": "open-street-map",
        "center": {
            "lat": center_latitude,
            "lon": center_longitude,
        },
        "zoom": 7,
    },
    height=680,
    margin={
        "r": 0,
        "t": 20,
        "l": 0,
        "b": 0,
    },
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "left",
        "x": 0,
    },
)


st.plotly_chart(
    figure,
    width="stretch",
)


tab_errors, tab_points, tab_explanation = st.tabs(
    [
        "Ошибки по траектории",
        "Координаты",
        "Как работает прогноз",
    ]
)


with tab_errors:
    error_data = trajectory_data[
        [
            "timestamp",
            "catboost_error_km",
            "baseline_error_km",
        ]
    ].copy()

    error_data["CatBoost"] = (
        error_data["catboost_error_km"] * 1000
    )

    error_data["Constant Velocity"] = (
        error_data["baseline_error_km"] * 1000
    )

    long_errors = error_data.melt(
        id_vars="timestamp",
        value_vars=[
            "CatBoost",
            "Constant Velocity",
        ],
        var_name="Модель",
        value_name="Ошибка, метры",
    )

    error_figure = px.line(
        long_errors,
        x="timestamp",
        y="Ошибка, метры",
        color="Модель",
        labels={
            "timestamp": "Время",
        },
    )

    st.plotly_chart(
        error_figure,
        width="stretch",
    )


with tab_points:
    coordinates = pd.DataFrame(
        {
            "Точка": [
                "Текущая позиция",
                "Прогноз CatBoost",
                "Фактическая позиция",
            ],
            "Широта": [
                selected_row["latitude"],
                selected_row["predicted_latitude"],
                selected_row["actual_latitude"],
            ],
            "Долгота": [
                selected_row["longitude"],
                selected_row["predicted_longitude"],
                selected_row["actual_longitude"],
            ],
        }
    )

    st.dataframe(
        coordinates,
        width="stretch",
        hide_index=True,
    )


with tab_explanation:
    st.markdown(
        """
        **Порядок получения прогноза:**

        1. Система получает текущие координаты, скорость и курс судна.
        2. Курс преобразуется в `course_sin` и `course_cos`.
        3. CatBoost прогнозирует изменение широты и долготы.
        4. Прогнозируемое смещение прибавляется к текущим координатам.
        5. Полученная позиция сравнивается с фактическим положением.
        """
    )

    st.info(
        "Baseline предполагает сохранение последнего направления "
        "и величины перемещения. CatBoost дополнительно учитывает "
        "курс, скорость, изменение движения и пространственные признаки."
    )


st.success(
    "CatBoost уменьшил среднюю ошибку относительно baseline "
    "на всех исследованных горизонтах."
)




