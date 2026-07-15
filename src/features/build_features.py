import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def get_group_column(data):
    if "trajectory_id" in data.columns:
        return "trajectory_id"

    if "mmsi" in data.columns:
        return "mmsi"

    return None


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
        2 * np.arctan2(
            np.sqrt(value),
            np.sqrt(1 - value),
        )
    )


def angular_difference(current_angle, previous_angle):
    return (
        (
            pd.to_numeric(current_angle, errors="coerce")
            - pd.to_numeric(previous_angle, errors="coerce")
            + 180
        )
        % 360
        - 180
    ).abs()


def build_navigation_features(data):
    result = data.copy()

    group_column = get_group_column(result)

    sort_columns = []

    if group_column:
        sort_columns.append(group_column)

    sort_columns.append("timestamp")

    result = result.sort_values(sort_columns).reset_index(drop=True)

    if group_column:
        grouped = result.groupby(
            group_column,
            sort=False,
        )

        previous_latitude = grouped["latitude"].shift(1)
        previous_longitude = grouped["longitude"].shift(1)
        previous_timestamp = grouped["timestamp"].shift(1)
    else:
        previous_latitude = result["latitude"].shift(1)
        previous_longitude = result["longitude"].shift(1)
        previous_timestamp = result["timestamp"].shift(1)

    result["previous_latitude"] = previous_latitude
    result["previous_longitude"] = previous_longitude
    result["previous_timestamp"] = previous_timestamp

    result["time_delta_seconds"] = (
        result["timestamp"] - previous_timestamp
    ).dt.total_seconds()

    result["distance_from_previous_km"] = haversine_distance_km(
        previous_latitude,
        previous_longitude,
        result["latitude"],
        result["longitude"],
    )

    time_delta_hours = result["time_delta_seconds"] / 3600

    valid_time = time_delta_hours.where(time_delta_hours > 0)

    result["calculated_speed_kmh"] = (
        result["distance_from_previous_km"] / valid_time
    )

    if "sog" in result.columns:
        if group_column:
            previous_speed = result.groupby(
                group_column,
                sort=False,
            )["sog"].shift(1)
        else:
            previous_speed = result["sog"].shift(1)

        result["speed_change"] = (
            result["sog"] - previous_speed
        ).abs()

    if "cog" in result.columns:
        if group_column:
            previous_course = result.groupby(
                group_column,
                sort=False,
            )["cog"].shift(1)
        else:
            previous_course = result["cog"].shift(1)

        result["course_change"] = angular_difference(
            result["cog"],
            previous_course,
        )

    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = result["timestamp"].dt.dayofweek

    return result
