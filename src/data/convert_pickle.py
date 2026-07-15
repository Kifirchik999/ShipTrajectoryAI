from pathlib import Path
import pickle

import pandas as pd


DATA_DIRECTORY = Path("data/raw")
OUTPUT_DIRECTORY = Path("data/processed")


def read_trajectories(file_path: Path) -> list[pd.DataFrame]:
    trajectories = []

    with file_path.open("rb") as file:
        while True:
            try:
                item = pickle.load(file)
            except EOFError:
                break

            if isinstance(item, dict):
                frame = pd.DataFrame(item)
            elif isinstance(item, pd.DataFrame):
                frame = item.copy()
            else:
                continue

            if frame.empty:
                continue

            trajectories.append(frame)

    return trajectories


def main() -> None:
    files = list(DATA_DIRECTORY.glob("*.pkl"))

    if not files:
        print("PKL-файлы не найдены")
        return

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        print("=" * 80)
        print(f"Файл: {file_path.name}")
        print(f"Размер: {file_path.stat().st_size / 1024 / 1024:.2f} МБ")
        print("Чтение траекторий...")

        trajectories = read_trajectories(file_path)

        if not trajectories:
            print("Траектории не найдены")
            continue

        lengths = [len(frame) for frame in trajectories]
        total_points = sum(lengths)

        vessel_ids = set()

        for frame in trajectories:
            if "mmsi" in frame.columns:
                vessel_ids.update(frame["mmsi"].dropna().unique())

        print(f"Количество траекторий: {len(trajectories)}")
        print(f"Количество AIS-точек: {total_points}")
        print(f"Количество уникальных судов: {len(vessel_ids)}")
        print(f"Минимальная длина траектории: {min(lengths)}")
        print(f"Средняя длина траектории: {sum(lengths) / len(lengths):.2f}")
        print(f"Максимальная длина траектории: {max(lengths)}")

        combined = []

        for trajectory_id, frame in enumerate(trajectories):
            current = frame.copy()
            current["trajectory_id"] = trajectory_id
            combined.append(current)

        data = pd.concat(
            combined,
            ignore_index=True,
        )

        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(
                data["timestamp"],
                unit="s",
                errors="coerce",
                utc=True,
            )

        output_path = OUTPUT_DIRECTORY / "ais_trajectories.csv"

        data.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Итоговая таблица: {data.shape}")
        print(f"Сохранено в: {output_path}")

        print("\nПервые строки:")
        print(data.head().to_string(index=False))


if __name__ == "__main__":
    main()
