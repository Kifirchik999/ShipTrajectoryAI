from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


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
MODEL_PATH = Path("models/isolation_forest.joblib")
SCALER_PATH = Path("models/isolation_scaler.joblib")
OUTPUT_PATH = Path("reports/tables/isolation_forest_results.csv")


FEATURES = [
    "sog",
    "speed_change",
    "course_change",
    "distance_from_previous_km",
    "calculated_speed_kmh",
    "time_delta_seconds",
]


def main():
    data = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    data = standardize_column_names(data)
    validate_required_columns(data)
    data = clean_ais_data(data)
    data = build_navigation_features(data)

    model_data = data.dropna(
        subset=FEATURES
    ).copy()

    X = model_data[FEATURES]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )

    predictions = model.fit_predict(X_scaled)

    model_data["isolation_label"] = predictions

    model_data["isolation_anomaly"] = (
        model_data["isolation_label"] == -1
    )

    model_data["anomaly_score"] = (
        -model.decision_function(X_scaled)
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    result_columns = [
        "trajectory_id",
        "mmsi",
        "timestamp",
        "latitude",
        "longitude",
        "sog",
        "cog",
        "speed_change",
        "course_change",
        "distance_from_previous_km",
        "calculated_speed_kmh",
        "time_delta_seconds",
        "isolation_anomaly",
        "anomaly_score",
    ]

    result_columns = [
        column
        for column in result_columns
        if column in model_data.columns
    ]

    model_data[result_columns].to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    anomaly_count = int(
        model_data["isolation_anomaly"].sum()
    )

    anomaly_share = (
        model_data["isolation_anomaly"].mean()
        * 100
    )

    print(f"Количество точек: {len(model_data)}")
    print(f"Обнаружено аномалий: {anomaly_count}")
    print(f"Доля аномалий: {anomaly_share:.2f}%")

    print()
    print("Средние значения признаков")

    comparison = (
        model_data.groupby("isolation_anomaly")[FEATURES]
        .mean()
        .T
    )

    comparison.columns = [
        "normal",
        "anomaly",
    ]

    print(
        comparison.round(4).to_string()
    )

    print()
    print("Наиболее подозрительные точки")

    top_anomalies = (
        model_data[
            model_data["isolation_anomaly"]
        ]
        .sort_values(
            "anomaly_score",
            ascending=False,
        )
        .head(10)
    )

    display_columns = [
        column
        for column in [
            "trajectory_id",
            "mmsi",
            "timestamp",
            "sog",
            "course_change",
            "calculated_speed_kmh",
            "anomaly_score",
        ]
        if column in top_anomalies.columns
    ]

    print(
        top_anomalies[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print()
    print(f"Модель сохранена: {MODEL_PATH}")
    print(f"Масштабировщик сохранён: {SCALER_PATH}")
    print(f"Результаты сохранены: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
