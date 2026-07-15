from pathlib import Path
import sys

import pandas as pd


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


INPUT_PATH = Path("data/processed/ais_sample.csv")


def main():
    data = pd.read_csv(INPUT_PATH)

    data = standardize_column_names(data)
    validate_required_columns(data)

    data = clean_ais_data(data)
    data = build_navigation_features(data)
    data = detect_rule_based_anomalies(data)

    anomaly_columns = [
        "speed_anomaly",
        "course_anomaly",
        "position_jump_anomaly",
        "time_gap_anomaly",
    ]

    print(f"Количество строк: {len(data)}")
    print(f"Количество аномалий: {int(data['is_anomaly'].sum())}")
    print(f"Доля аномалий: {data['is_anomaly'].mean() * 100:.2f}%")

    print("\nКоличество срабатываний по правилам:")

    for column in anomaly_columns:
        print(f"{column}: {int(data[column].sum())}")

    print("\nРаспределение причин:")

    print(
        data.loc[data["is_anomaly"], "anomaly_reason"]
        .value_counts()
        .to_string()
    )

    print("\nСтатистика изменения курса:")

    print(
        data["course_change"]
        .describe(
            percentiles=[
                0.90,
                0.95,
                0.97,
                0.99,
            ]
        )
    )

    print("\nСтатистика рассчитанной скорости:")

    print(
        data["calculated_speed_kmh"]
        .replace([float("inf"), float("-inf")], pd.NA)
        .dropna()
        .describe(
            percentiles=[
                0.90,
                0.95,
                0.97,
                0.99,
            ]
        )
    )


if __name__ == "__main__":
    main()
