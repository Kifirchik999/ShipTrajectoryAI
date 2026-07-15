from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


INPUT_PATH = Path("data/processed/ais_sample.csv")
OUTPUT_PATH = Path("reports/tables/baseline_metrics.csv")
EARTH_RADIUS_KM = 6371.0088


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


def main():
    data = pd.read_csv(INPUT_PATH)

    data = standardize_column_names(data)
    validate_required_columns(data)
    data = clean_ais_data(data)
    data = build_navigation_features(data)
    data = predict_next_position_constant_velocity(data)
    data = attach_actual_next_position(data)

    data["prediction_error_km"] = haversine_distance_km(
        data["predicted_next_latitude"],
        data["predicted_next_longitude"],
        data["actual_next_latitude"],
        data["actual_next_longitude"],
    )

    valid = data.dropna(
        subset=[
            "prediction_error_km",
            "trajectory_id",
        ]
    ).copy()

    trajectory_metrics = (
        valid.groupby("trajectory_id")
        .agg(
            mmsi=("mmsi", "first"),
            points=("prediction_error_km", "count"),
            ade_km=("prediction_error_km", "mean"),
            median_error_km=("prediction_error_km", "median"),
            max_error_km=("prediction_error_km", "max"),
            fde_km=("prediction_error_km", "last"),
        )
        .reset_index()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_metrics.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Количество оценённых точек: {len(valid)}")
    print(
        f"Количество траекторий: "
        f"{trajectory_metrics['trajectory_id'].nunique()}"
    )

    print("\nОбщие метрики baseline:")

    print(
        f"Средняя ошибка: "
        f"{valid['prediction_error_km'].mean():.4f} км"
    )

    print(
        f"Медианная ошибка: "
        f"{valid['prediction_error_km'].median():.4f} км"
    )

    print(
        f"RMSE: "
        f"{np.sqrt(np.mean(valid['prediction_error_km'] ** 2)):.4f} км"
    )

    print(
        f"90-й процентиль: "
        f"{valid['prediction_error_km'].quantile(0.90):.4f} км"
    )

    print(
        f"95-й процентиль: "
        f"{valid['prediction_error_km'].quantile(0.95):.4f} км"
    )

    print(
        f"Средний ADE: "
        f"{trajectory_metrics['ade_km'].mean():.4f} км"
    )

    print(
        f"Средний FDE: "
        f"{trajectory_metrics['fde_km'].mean():.4f} км"
    )

    print("\nЛучшие траектории по ADE:")

    print(
        trajectory_metrics
        .sort_values("ade_km")
        .head(5)
        .to_string(index=False)
    )

    print("\nХудшие траектории по ADE:")

    print(
        trajectory_metrics
        .sort_values("ade_km", ascending=False)
        .head(5)
        .to_string(index=False)
    )

    print(f"\nМетрики сохранены в: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
