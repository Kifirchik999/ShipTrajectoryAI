from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.loader import (
    standardize_column_names,
    validate_required_columns,
)
from src.features.build_features import build_navigation_features
from src.preprocessing.cleaning import clean_ais_data


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ais_sample.csv"

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "catboost_multihorizon"
    / "horizon_3"
)

LATITUDE_MODEL_PATH = MODEL_DIRECTORY / "latitude.cbm"
LONGITUDE_MODEL_PATH = MODEL_DIRECTORY / "longitude.cbm"

FIGURES_DIRECTORY = PROJECT_ROOT / "reports" / "figures"
TABLES_DIRECTORY = PROJECT_ROOT / "reports" / "tables"

SUMMARY_PATH = FIGURES_DIRECTORY / "shap_summary_horizon_3.png"
BAR_PATH = FIGURES_DIRECTORY / "shap_importance_horizon_3.png"
TABLE_PATH = TABLES_DIRECTORY / "shap_importance_horizon_3.csv"

RANDOM_STATE = 42
SAMPLE_SIZE = 1500


FEATURES = [
    "latitude",
    "longitude",
    "sog",
    "course_sin",
    "course_cos",
    "speed_change",
    "course_change",
    "distance_from_previous_km",
    "calculated_speed_kmh",
    "time_delta_seconds",
    "hour_sin",
    "hour_cos",
    "day_of_week",
]


FEATURE_NAMES_RU = {
    "latitude": "Широта",
    "longitude": "Долгота",
    "sog": "Скорость SOG",
    "course_sin": "Курс: sin",
    "course_cos": "Курс: cos",
    "speed_change": "Изменение скорости",
    "course_change": "Изменение курса",
    "distance_from_previous_km": "Расстояние от прошлой точки",
    "calculated_speed_kmh": "Расчётная скорость",
    "time_delta_seconds": "Интервал времени",
    "hour_sin": "Час: sin",
    "hour_cos": "Час: cos",
    "day_of_week": "День недели",
}


def prepare_data():
    data = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    data = standardize_column_names(data)
    validate_required_columns(data)
    data = clean_ais_data(data)

    data = data.sort_values(
        ["trajectory_id", "timestamp"]
    ).reset_index(drop=True)

    data = build_navigation_features(data)

    course_radians = np.radians(data["cog"])

    data["course_sin"] = np.sin(course_radians)
    data["course_cos"] = np.cos(course_radians)

    hour_radians = 2 * np.pi * data["hour"] / 24

    data["hour_sin"] = np.sin(hour_radians)
    data["hour_cos"] = np.cos(hour_radians)

    data = data.dropna(
        subset=FEATURES
    ).copy()

    return data


def load_model(path):
    model = CatBoostRegressor()
    model.load_model(str(path))

    return model


def calculate_shap_values(model, features):
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(features)

    return np.asarray(values)


def main():
    if not LATITUDE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Не найдена модель широты: {LATITUDE_MODEL_PATH}"
        )

    if not LONGITUDE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Не найдена модель долготы: {LONGITUDE_MODEL_PATH}"
        )

    FIGURES_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = prepare_data()

    sample_size = min(
        SAMPLE_SIZE,
        len(data),
    )

    sample = data.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    ).copy()

    X = sample[FEATURES].copy()

    latitude_model = load_model(
        LATITUDE_MODEL_PATH
    )

    longitude_model = load_model(
        LONGITUDE_MODEL_PATH
    )

    print("Расчёт SHAP для модели широты")

    latitude_shap = calculate_shap_values(
        latitude_model,
        X,
    )

    print("Расчёт SHAP для модели долготы")

    longitude_shap = calculate_shap_values(
        longitude_model,
        X,
    )

    combined_shap = (
        np.abs(latitude_shap)
        + np.abs(longitude_shap)
    ) / 2

    signed_combined_shap = (
        latitude_shap
        + longitude_shap
    ) / 2

    mean_importance = combined_shap.mean(
        axis=0
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "feature_ru": [
                FEATURE_NAMES_RU[feature]
                for feature in FEATURES
            ],
            "mean_absolute_shap": mean_importance,
        }
    ).sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance.to_csv(
        TABLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    display_X = X.rename(
        columns=FEATURE_NAMES_RU
    )

    plt.figure()

    shap.summary_plot(
        signed_combined_shap,
        display_X,
        show=False,
        max_display=13,
    )

    plt.title(
        "Влияние признаков на прогноз CatBoost",
        pad=20,
    )

    plt.tight_layout()

    plt.savefig(
        SUMMARY_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    plot_data = importance.sort_values(
        "mean_absolute_shap"
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.barh(
        plot_data["feature_ru"],
        plot_data["mean_absolute_shap"],
    )

    axis.set_title(
        "Глобальная важность признаков по SHAP"
    )

    axis.set_xlabel(
        "Средний абсолютный SHAP-вклад"
    )

    axis.set_ylabel(
        "Признак"
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        BAR_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print()
    print("Главные признаки:")

    print(
        importance.head(10)
        .round(6)
        .to_string(index=False)
    )

    print()
    print(f"SHAP summary: {SUMMARY_PATH}")
    print(f"SHAP importance: {BAR_PATH}")
    print(f"Таблица: {TABLE_PATH}")


if __name__ == "__main__":
    main()
