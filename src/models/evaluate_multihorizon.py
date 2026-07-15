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
from src.preprocessing.cleaning import clean_ais_data


INPUT_PATH = Path("data/processed/ais_sample.csv")
OUTPUT_PATH = Path("reports/tables/multihorizon_baseline.csv")
EARTH_RADIUS_KM = 6371.0088
HORIZONS = [1, 3, 5, 10]


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


def evaluate_horizon(data, horizon):
    result = data.copy()

    grouped = result.groupby(
        "trajectory_id",
        sort=False,
    )

    previous_latitude = grouped["latitude"].shift(1)
    previous_longitude = grouped["longitude"].shift(1)

    latitude_delta = result["latitude"] - previous_latitude
    longitude_delta = result["longitude"] - previous_longitude

    result["predicted_latitude"] = (
        result["latitude"]
        + latitude_delta * horizon
    )

    result["predicted_longitude"] = (
        result["longitude"]
        + longitude_delta * horizon
    )

    result["actual_latitude"] = grouped[
        "latitude"
    ].shift(-horizon)

    result["actual_longitude"] = grouped[
        "longitude"
    ].shift(-horizon)

    result["error_km"] = haversine_distance_km(
        result["predicted_latitude"],
        result["predicted_longitude"],
        result["actual_latitude"],
        result["actual_longitude"],
    )

    valid = result.dropna(
        subset=["error_km"]
    ).copy()

    trajectory_metrics = (
        valid.groupby("trajectory_id")
        .agg(
            ade_km=("error_km", "mean"),
            fde_km=("error_km", "last"),
        )
        .reset_index()
    )

    return {
        "horizon_steps": horizon,
        "evaluated_points": len(valid),
        "mean_error_km": valid["error_km"].mean(),
        "median_error_km": valid["error_km"].median(),
        "rmse_km": np.sqrt(
            np.mean(valid["error_km"] ** 2)
        ),
        "p90_error_km": valid["error_km"].quantile(0.90),
        "p95_error_km": valid["error_km"].quantile(0.95),
        "mean_ade_km": trajectory_metrics["ade_km"].mean(),
        "mean_fde_km": trajectory_metrics["fde_km"].mean(),
    }


def main():
    data = pd.read_csv(INPUT_PATH)

    data = standardize_column_names(data)
    validate_required_columns(data)
    data = clean_ais_data(data)

    data = data.sort_values(
        ["trajectory_id", "timestamp"]
    ).reset_index(drop=True)

    results = []

    for horizon in HORIZONS:
        metrics = evaluate_horizon(data, horizon)
        results.append(metrics)

        print("=" * 60)
        print(f"Горизонт: {horizon} шагов")
        print(
            f"Средняя ошибка: "
            f"{metrics['mean_error_km']:.4f} км"
        )
        print(
            f"Медианная ошибка: "
            f"{metrics['median_error_km']:.4f} км"
        )
        print(
            f"RMSE: "
            f"{metrics['rmse_km']:.4f} км"
        )
        print(
            f"Средний ADE: "
            f"{metrics['mean_ade_km']:.4f} км"
        )
        print(
            f"Средний FDE: "
            f"{metrics['mean_fde_km']:.4f} км"
        )

    metrics_table = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_table.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Результаты сохранены в: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
