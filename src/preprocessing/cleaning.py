import pandas as pd


def clean_ais_data(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    numeric_columns = [
        "latitude",
        "longitude",
        "sog",
        "cog",
        "heading",
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
        utc=True,
    )

    result = result.dropna(
        subset=[
            "timestamp",
            "latitude",
            "longitude",
        ]
    )

    result = result[
        result["latitude"].between(-90, 90)
        & result["longitude"].between(-180, 180)
    ]

    if "sog" in result.columns:
        result = result[
            result["sog"].isna()
            | result["sog"].between(0, 80)
        ]

    if "cog" in result.columns:
        result.loc[
            ~result["cog"].between(0, 360),
            "cog",
        ] = pd.NA

    if "heading" in result.columns:
        result.loc[
            ~result["heading"].between(0, 360),
            "heading",
        ] = pd.NA

    duplicate_columns = [
        column
        for column in [
            "mmsi",
            "timestamp",
            "latitude",
            "longitude",
        ]
        if column in result.columns
    ]

    result = result.drop_duplicates(
        subset=duplicate_columns,
        keep="first",
    )

    sort_columns = [
        column
        for column in ["mmsi", "timestamp"]
        if column in result.columns
    ]

    result = result.sort_values(sort_columns)

    return result.reset_index(drop=True)
