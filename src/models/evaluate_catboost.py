from pathlib import Path
import sys

import numpy as np
import pandas as pd
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


INPUT_PATH = Path("data/processed/ais_sample.csv")
MODELS_DIRECTORY = Path("models/catboost_multihorizon")
TABLES_DIRECTORY = Path("reports/tables")

COMPARISON_PATH = TABLES_DIRECTORY / "model_comparison.csv"
IMPORTANCE_PATH = TABLES_DIRECTORY / "catboost_feature_importance.csv"
PREDICTIONS_PATH = TABLES_DIRECTORY / "catboost_predictions.csv"

HORIZONS = [1, 3, 5, 10]
EARTH_RADIUS_KM = 6371.0088
RANDOM_STATE = 42


def haversine_distance_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon1 = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat2 = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon2 = np.radians(pd.to_numeric(lon2, errors="coerce"))

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    value = np.clip(value, 0, 1)

    return EARTH_RADIUS_KM * (
        2
        * np.arctan2(
            np.sqrt(value),
            np.sqrt(1 - value),
        )
    )


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

    return data


def split_trajectories(data):
    trajectory_ids = data["trajectory_id"].dropna().unique()

    random_generator = np.random.default_rng(RANDOM_STATE)
    random_generator.shuffle(trajectory_ids)

    split_index = int(len(trajectory_ids) * 0.8)

    train_ids = set(trajectory_ids[:split_index])
    test_ids = set(trajectory_ids[split_index:])

    return train_ids, test_ids


def calculate_metrics(errors, trajectory_ids):
    metric_data = pd.DataFrame(
        {
            "error_km": errors,
            "trajectory_id": trajectory_ids,
        }
    )

    trajectory_ade = (
        metric_data.groupby("trajectory_id")["error_km"]
        .mean()
    )

    trajectory_fde = (
        metric_data.groupby("trajectory_id")["error_km"]
        .last()
    )

    return {
        "evaluated_points": len(metric_data),
        "mean_error_km": metric_data["error_km"].mean(),
        "median_error_km": metric_data["error_km"].median(),
        "rmse_km": np.sqrt(
            np.mean(metric_data["error_km"] ** 2)
        ),
        "p90_error_km": metric_data["error_km"].quantile(0.90),
        "p95_error_km": metric_data["error_km"].quantile(0.95),
        "mean_ade_km": trajectory_ade.mean(),
        "mean_fde_km": trajectory_fde.mean(),
    }


def create_model():
    return CatBoostRegressor(
        iterations=700,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        early_stopping_rounds=60,
        verbose=100,
        allow_writing_files=False,
    )


def main():
    data = prepare_data()

    train_ids, test_ids = split_trajectories(data)

    print(f"Всего траекторий: {data['trajectory_id'].nunique()}")
    print(f"Обучающих траекторий: {len(train_ids)}")
    print(f"Тестовых траекторий: {len(test_ids)}")

    features = [
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

    MODELS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TABLES_DIRECTORY.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    importance_rows = []
    prediction_tables = []

    for horizon in HORIZONS:
        print()
        print("=" * 70)
        print(f"Горизонт прогнозирования: {horizon} шагов")

        horizon_data = data.copy()

        grouped = horizon_data.groupby(
            "trajectory_id",
            sort=False,
        )

        horizon_data["actual_latitude"] = grouped[
            "latitude"
        ].shift(-horizon)

        horizon_data["actual_longitude"] = grouped[
            "longitude"
        ].shift(-horizon)

        horizon_data["target_delta_latitude"] = (
            horizon_data["actual_latitude"]
            - horizon_data["latitude"]
        )

        horizon_data["target_delta_longitude"] = (
            horizon_data["actual_longitude"]
            - horizon_data["longitude"]
        )

        required_columns = (
            features
            + [
                "target_delta_latitude",
                "target_delta_longitude",
                "actual_latitude",
                "actual_longitude",
                "previous_latitude",
                "previous_longitude",
            ]
        )

        horizon_data = horizon_data.dropna(
            subset=required_columns
        ).copy()

        train_data = horizon_data[
            horizon_data["trajectory_id"].isin(train_ids)
        ].copy()

        test_data = horizon_data[
            horizon_data["trajectory_id"].isin(test_ids)
        ].copy()

        X_train = train_data[features]
        X_test = test_data[features]

        y_latitude_train = train_data[
            "target_delta_latitude"
        ]

        y_longitude_train = train_data[
            "target_delta_longitude"
        ]

        y_latitude_test = test_data[
            "target_delta_latitude"
        ]

        y_longitude_test = test_data[
            "target_delta_longitude"
        ]

        latitude_model = create_model()
        longitude_model = create_model()

        print("Обучение модели широты")

        latitude_model.fit(
            X_train,
            y_latitude_train,
            eval_set=(X_test, y_latitude_test),
        )

        print("Обучение модели долготы")

        longitude_model.fit(
            X_train,
            y_longitude_train,
            eval_set=(X_test, y_longitude_test),
        )

        horizon_directory = (
            MODELS_DIRECTORY / f"horizon_{horizon}"
        )

        horizon_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        latitude_model.save_model(
            str(horizon_directory / "latitude.cbm")
        )

        longitude_model.save_model(
            str(horizon_directory / "longitude.cbm")
        )

        predicted_delta_latitude = latitude_model.predict(
            X_test
        )

        predicted_delta_longitude = longitude_model.predict(
            X_test
        )

        predicted_latitude = (
            test_data["latitude"].to_numpy()
            + predicted_delta_latitude
        )

        predicted_longitude = (
            test_data["longitude"].to_numpy()
            + predicted_delta_longitude
        )

        catboost_errors = haversine_distance_km(
            predicted_latitude,
            predicted_longitude,
            test_data["actual_latitude"],
            test_data["actual_longitude"],
        )

        latitude_step = (
            test_data["latitude"]
            - test_data["previous_latitude"]
        )

        longitude_step = (
            test_data["longitude"]
            - test_data["previous_longitude"]
        )

        baseline_latitude = (
            test_data["latitude"]
            + latitude_step * horizon
        )

        baseline_longitude = (
            test_data["longitude"]
            + longitude_step * horizon
        )

        baseline_errors = haversine_distance_km(
            baseline_latitude,
            baseline_longitude,
            test_data["actual_latitude"],
            test_data["actual_longitude"],
        )

        catboost_metrics = calculate_metrics(
            catboost_errors,
            test_data["trajectory_id"].to_numpy(),
        )

        baseline_metrics = calculate_metrics(
            baseline_errors,
            test_data["trajectory_id"].to_numpy(),
        )

        comparison_rows.append(
            {
                "model": "Constant Velocity",
                "horizon_steps": horizon,
                **baseline_metrics,
            }
        )

        comparison_rows.append(
            {
                "model": "CatBoost",
                "horizon_steps": horizon,
                **catboost_metrics,
            }
        )

        latitude_importance = (
            latitude_model.get_feature_importance()
        )

        longitude_importance = (
            longitude_model.get_feature_importance()
        )

        average_importance = (
            latitude_importance
            + longitude_importance
        ) / 2

        for feature, importance in zip(
            features,
            average_importance,
        ):
            importance_rows.append(
                {
                    "horizon_steps": horizon,
                    "feature": feature,
                    "importance": importance,
                }
            )

        predictions = test_data[
            [
                "trajectory_id",
                "mmsi",
                "timestamp",
                "latitude",
                "longitude",
                "actual_latitude",
                "actual_longitude",
            ]
        ].copy()

        predictions["horizon_steps"] = horizon
        predictions["predicted_latitude"] = predicted_latitude
        predictions["predicted_longitude"] = predicted_longitude
        predictions["catboost_error_km"] = np.asarray(
            catboost_errors
        )
        predictions["baseline_error_km"] = np.asarray(
            baseline_errors
        )

        prediction_tables.append(predictions)

        print()
        print(
            f"Baseline, средняя ошибка: "
            f"{baseline_metrics['mean_error_km']:.4f} км"
        )

        print(
            f"CatBoost, средняя ошибка: "
            f"{catboost_metrics['mean_error_km']:.4f} км"
        )

        improvement = (
            (
                baseline_metrics["mean_error_km"]
                - catboost_metrics["mean_error_km"]
            )
            / baseline_metrics["mean_error_km"]
            * 100
        )

        print(
            f"Изменение относительно baseline: "
            f"{improvement:.2f}%"
        )

    comparison = pd.DataFrame(comparison_rows)

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    importance = pd.DataFrame(importance_rows)

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("Итоговое сравнение моделей")
    print()

    display_columns = [
        "model",
        "horizon_steps",
        "mean_error_km",
        "median_error_km",
        "rmse_km",
        "mean_ade_km",
        "mean_fde_km",
    ]

    print(
        comparison[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print()
    print(f"Сравнение сохранено: {COMPARISON_PATH}")
    print(f"Важность признаков: {IMPORTANCE_PATH}")
    print(f"Прогнозы сохранены: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
