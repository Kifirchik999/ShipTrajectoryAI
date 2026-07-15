from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


COMPARISON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "model_comparison.csv"
)

IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "catboost_feature_importance.csv"
)


st.set_page_config(
    page_title="Сравнение моделей",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Сравнение моделей")

st.caption(
    "Сравнение CatBoost и модели постоянной скорости "
    "на разных горизонтах прогнозирования"
)


if not COMPARISON_PATH.exists():
    st.error(
        "Файл model_comparison.csv не найден. "
        "Сначала запустите evaluate_catboost.py"
    )
    st.stop()


comparison = pd.read_csv(COMPARISON_PATH)

comparison["mean_error_m"] = (
    comparison["mean_error_km"] * 1000
)

comparison["median_error_m"] = (
    comparison["median_error_km"] * 1000
)

comparison["rmse_m"] = (
    comparison["rmse_km"] * 1000
)


baseline = (
    comparison[
        comparison["model"] == "Constant Velocity"
    ]
    .set_index("horizon_steps")
)

catboost = (
    comparison[
        comparison["model"] == "CatBoost"
    ]
    .set_index("horizon_steps")
)

summary = pd.DataFrame(
    {
        "Горизонт": baseline.index,
        "Baseline, м": baseline["mean_error_m"].values,
        "CatBoost, м": catboost["mean_error_m"].values,
    }
)

summary["Улучшение, %"] = (
    (
        summary["Baseline, м"]
        - summary["CatBoost, м"]
    )
    / summary["Baseline, м"]
    * 100
)


best_row = summary.loc[
    summary["Улучшение, %"].idxmax()
]

metric_1, metric_2, metric_3 = st.columns(3)

metric_1.metric(
    "Ошибка CatBoost, 1 шаг",
    f"{summary.iloc[0]['CatBoost, м']:.1f} м",
)

metric_2.metric(
    "Максимальное улучшение",
    f"{best_row['Улучшение, %']:.1f}%",
)

metric_3.metric(
    "Лучший горизонт",
    f"{int(best_row['Горизонт'])} шага",
)


st.subheader("Средняя ошибка прогнозирования")

comparison_figure = px.bar(
    comparison,
    x="horizon_steps",
    y="mean_error_m",
    color="model",
    barmode="group",
    labels={
        "horizon_steps": "Горизонт прогнозирования, AIS-шаги",
        "mean_error_m": "Средняя ошибка, метры",
        "model": "Модель",
    },
)

comparison_figure.update_layout(
    xaxis={
        "type": "category",
    }
)

st.plotly_chart(
    comparison_figure,
    width="stretch",
)


st.subheader("Улучшение CatBoost относительно baseline")

improvement_figure = px.bar(
    summary,
    x="Горизонт",
    y="Улучшение, %",
    text_auto=".1f",
)

improvement_figure.update_layout(
    xaxis={
        "type": "category",
        "title": "Горизонт прогнозирования, AIS-шаги",
    },
    yaxis_title="Снижение средней ошибки, %",
)

st.plotly_chart(
    improvement_figure,
    width="stretch",
)


st.subheader("Итоговая таблица")

display_summary = summary.copy()

display_summary["Baseline, м"] = (
    display_summary["Baseline, м"].round(1)
)

display_summary["CatBoost, м"] = (
    display_summary["CatBoost, м"].round(1)
)

display_summary["Улучшение, %"] = (
    display_summary["Улучшение, %"].round(1)
)

st.dataframe(
    display_summary,
    width="stretch",
    hide_index=True,
)


if IMPORTANCE_PATH.exists():
    st.subheader("Важность навигационных признаков")

    importance = pd.read_csv(IMPORTANCE_PATH)

    average_importance = (
        importance.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    importance_figure = px.bar(
        average_importance,
        x="importance",
        y="feature",
        orientation="h",
        labels={
            "importance": "Важность признака",
            "feature": "Признак",
        },
    )

    importance_figure.update_layout(
        yaxis={
            "categoryorder": "total ascending",
        }
    )

    st.plotly_chart(
        importance_figure,
        width="stretch",
    )

    st.info(
        "Признаки course_sin и course_cos вместе представляют "
        "направление движения судна. Их высокая важность показывает, "
        "что курс является главным фактором прогноза."
    )


st.subheader("Основной вывод")

st.success(
    "CatBoost превзошёл модель постоянной скорости на всех "
    "исследованных горизонтах. Максимальное снижение средней "
    "ошибки достигнуто при прогнозировании на 3 AIS-шага."
)



