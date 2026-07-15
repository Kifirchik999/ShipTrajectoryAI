from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPARISON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "model_comparison.csv"
)

SHAP_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "shap_summary_horizon_3.png"
)


st.set_page_config(
    page_title="О проекте",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 ShipTrajectoryAI")

st.subheader(
    "Интеллектуальная система прогнозирования траекторий "
    "судов и обнаружения аномального навигационного поведения"
)

st.markdown(
    """
    **ShipTrajectoryAI** анализирует исторические AIS-сообщения,
    формирует отдельные траектории судов, прогнозирует будущие
    координаты и выявляет необычные участки движения.
    """
)


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "AIS-точек",
    "2 152 999",
)

metric_2.metric(
    "Траекторий",
    "12 591",
)

metric_3.metric(
    "Судов",
    "2 078",
)

metric_4.metric(
    "Горизонты",
    "1, 3, 5, 10",
)


st.header("Цель проекта")

st.write(
    "Разработать понятную и интерпретируемую ML-систему, "
    "которая помогает анализировать движение судов, "
    "прогнозировать их дальнейшее положение и обнаруживать "
    "подозрительные изменения навигационного поведения."
)


st.header("Архитектура системы")

st.code(
    """
AIS-данные
    |
    v
Очистка и проверка данных
    |
    v
Формирование отдельных траекторий
    |
    v
Создание навигационных признаков
    |
    +-----------------------------+
    |                             |
    v                             v
Прогнозирование              Поиск аномалий
    |                             |
    + Constant Velocity           + Rule-based
    + CatBoost                    + Isolation Forest
    |                             |
    +-------------+---------------+
                  |
                  v
        Streamlit-интерфейс
    """,
    language="text",
)


st.header("Используемые методы")

column_1, column_2 = st.columns(2)

with column_1:
    st.subheader("Прогнозирование")

    st.markdown(
        """
        **Constant Velocity**

        Базовая модель предполагает, что судно продолжит
        движение с тем же смещением, что и на предыдущем шаге.

        **CatBoost**

        Прогнозирует изменение широты и долготы по скорости,
        курсу, предыдущему перемещению и другим признакам.
        """
    )

with column_2:
    st.subheader("Обнаружение аномалий")

    st.markdown(
        """
        **Rule-based detector**

        Использует понятные пороги для резкого изменения курса,
        высокой скорости, скачка координат и временного разрыва.

        **Isolation Forest**

        Находит редкие комбинации навигационных признаков
        без заранее заданных меток аномалий.
        """
    )


if COMPARISON_PATH.exists():
    comparison = pd.read_csv(
        COMPARISON_PATH
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
            "Горизонт, AIS-шаги": baseline.index,
            "Baseline, м": (
                baseline["mean_error_km"].values
                * 1000
            ),
            "CatBoost, м": (
                catboost["mean_error_km"].values
                * 1000
            ),
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

    st.header("Основные результаты")

    st.dataframe(
        summary.round(1),
        width="stretch",
        hide_index=True,
    )

    best_result = summary.loc[
        summary["Улучшение, %"].idxmax()
    ]

    st.success(
        "Максимальное снижение средней ошибки составило "
        f"{best_result['Улучшение, %']:.1f}% "
        f"на горизонте "
        f"{int(best_result['Горизонт, AIS-шаги'])} AIS-шага."
    )


st.header("Интерпретация модели")

st.write(
    "Анализ важности признаков и SHAP показал, что "
    "основное влияние на прогноз оказывают курс и скорость "
    "судна. Это соответствует физическому смыслу движения."
)

if SHAP_PATH.exists():
    st.image(
        str(SHAP_PATH),
        caption=(
            "SHAP-анализ модели CatBoost "
            "для горизонта 3 AIS-шага"
        ),
        width="stretch",
    )


st.header("Ограничения")

st.markdown(
    """
    - ошибка увеличивается при росте горизонта прогнозирования;
    - CatBoost лучше baseline в среднем, но может проигрывать
      на отдельных точках;
    - Isolation Forest обнаруживает статистические выбросы,
      которые не всегда означают опасную ситуацию;
    - используемый обучающий набор содержит исторические данные
      конкретного морского региона.
    """
)


st.header("Практический результат")

st.info(
    "Разработано работающее приложение, объединяющее "
    "обработку AIS-данных, прогнозирование траекторий, "
    "поиск аномалий, интерпретацию моделей и интерактивную "
    "визуализацию результатов."
)
