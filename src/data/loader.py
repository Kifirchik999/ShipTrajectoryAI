from pathlib import Path

import pandas as pd


COLUMN_ALIASES = {
    "mmsi": ["mmsi", "vessel_id", "ship_id", "imo"],
    "timestamp": [
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "base_date_time",
        "basedatetime",
    ],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng", "long"],
    "sog": ["sog", "speed", "speed_over_ground"],
    "cog": ["cog", "course", "course_over_ground"],
    "heading": ["heading", "true_heading"],
}


def normalize_column_name(column: str) -> str:
    return (
        column.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def standardize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    normalized_columns = {
        column: normalize_column_name(column)
        for column in result.columns
    }

    result = result.rename(columns=normalized_columns)

    rename_map = {}

    for target_name, aliases in COLUMN_ALIASES.items():
        if target_name in result.columns:
            continue

        for alias in aliases:
            if alias in result.columns:
                rename_map[alias] = target_name
                break

    return result.rename(columns=rename_map)


def load_ais_csv(
    file_path: str | Path,
    separator: str | None = None,
) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Поддерживаются только CSV-файлы")

    if separator is None:
        data = pd.read_csv(
            path,
            sep=None,
            engine="python",
            low_memory=False,
        )
    else:
        data = pd.read_csv(
            path,
            sep=separator,
            low_memory=False,
        )

    if data.empty:
        raise ValueError("CSV-файл не содержит данных")

    return standardize_column_names(data)


def validate_required_columns(data: pd.DataFrame) -> None:
    required_columns = {
        "timestamp",
        "latitude",
        "longitude",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))

        raise ValueError(
            f"Отсутствуют обязательные столбцы: {missing}"
        )
