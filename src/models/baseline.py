import numpy as np


def get_group_column(data):
    if "trajectory_id" in data.columns:
        return "trajectory_id"

    if "mmsi" in data.columns:
        return "mmsi"

    return None


def predict_next_position_constant_velocity(data):
    result = data.copy()

    group_column = get_group_column(result)

    if group_column:
        grouped = result.groupby(
            group_column,
            sort=False,
        )

        previous_latitude = grouped["latitude"].shift(1)
        previous_longitude = grouped["longitude"].shift(1)
    else:
        previous_latitude = result["latitude"].shift(1)
        previous_longitude = result["longitude"].shift(1)

    latitude_delta = result["latitude"] - previous_latitude
    longitude_delta = result["longitude"] - previous_longitude

    result["predicted_next_latitude"] = (
        result["latitude"] + latitude_delta
    )

    result["predicted_next_longitude"] = (
        result["longitude"] + longitude_delta
    )

    result.loc[
        result["predicted_next_latitude"].abs() > 90,
        "predicted_next_latitude",
    ] = np.nan

    result.loc[
        result["predicted_next_longitude"].abs() > 180,
        "predicted_next_longitude",
    ] = np.nan

    return result


def attach_actual_next_position(data):
    result = data.copy()

    group_column = get_group_column(result)

    if group_column:
        grouped = result.groupby(
            group_column,
            sort=False,
        )

        result["actual_next_latitude"] = grouped[
            "latitude"
        ].shift(-1)

        result["actual_next_longitude"] = grouped[
            "longitude"
        ].shift(-1)
    else:
        result["actual_next_latitude"] = result[
            "latitude"
        ].shift(-1)

        result["actual_next_longitude"] = result[
            "longitude"
        ].shift(-1)

    return result
