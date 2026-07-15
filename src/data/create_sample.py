from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/ais_trajectories.csv")
OUTPUT_PATH = Path("data/processed/ais_sample.csv")


def main() -> None:
    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["timestamp"],
        low_memory=False,
    )

    print(f"Всего строк: {len(data)}")
    print(f"Всего траекторий: {data['trajectory_id'].nunique()}")
    print(f"Всего судов: {data['mmsi'].nunique()}")

    valid_trajectories = (
        data.groupby("trajectory_id")
        .size()
        .loc[lambda values: values >= 50]
        .index
    )

    selected_trajectories = valid_trajectories[:100]

    sample = data[
        data["trajectory_id"].isin(selected_trajectories)
    ].copy()

    sample = sample.sort_values(
        ["trajectory_id", "timestamp"]
    )

    sample.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Строк в выборке: {len(sample)}")
    print(f"Траекторий в выборке: {sample['trajectory_id'].nunique()}")
    print(f"Судов в выборке: {sample['mmsi'].nunique()}")
    print(f"Сохранено в: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
