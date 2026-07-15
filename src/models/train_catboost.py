from pathlib import Path
import sys

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.loader import (
    standardize_column_names,
    validate_required_columns,
)
from src.preprocessing.cleaning import clean_ais_data
from src.features.build_features import build_navigation_features


INPUT_PATH = Path(
    "data/processed/ais_sample.csv"
)


MODEL_PATH = Path(
    "models/catboost_lat_lon"
)


def prepare_dataset(data):

    data = standardize_column_names(data)

    validate_required_columns(data)

    data = clean_ais_data(data)

    data = build_navigation_features(data)


    data = data.sort_values(
        [
            "trajectory_id",
            "timestamp",
        ]
    )


    grouped = data.groupby(
        "trajectory_id",
        sort=False,
    )


    data["next_latitude"] = grouped[
        "latitude"
    ].shift(-1)


    data["next_longitude"] = grouped[
        "longitude"
    ].shift(-1)


    data["delta_lat"] = (
        data["next_latitude"]
        -
        data["latitude"]
    )


    data["delta_lon"] = (
        data["next_longitude"]
        -
        data["longitude"]
    )


    data = data.dropna()


    return data



def main():

    data = pd.read_csv(
        INPUT_PATH
    )


    data = prepare_dataset(data)


    features = [
        "latitude",
        "longitude",
        "sog",
        "cog",
        "speed_change",
        "course_change",
        "distance_from_previous_km",
        "calculated_speed_kmh",
        "hour",
        "day_of_week",
    ]


    X = data[features]


    y_lat = data["delta_lat"]

    y_lon = data["delta_lon"]


    split = int(
        len(data) * 0.8
    )


    X_train = X.iloc[:split]
    X_test = X.iloc[split:]


    y_lat_train = y_lat.iloc[:split]
    y_lat_test = y_lat.iloc[split:]


    y_lon_train = y_lon.iloc[:split]
    y_lon_test = y_lon.iloc[split:]


    model_lat = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=100,
    )


    model_lon = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=100,
    )


    print(
        "Обучение широты"
    )

    model_lat.fit(
        X_train,
        y_lat_train,
    )


    print(
        "Обучение долготы"
    )

    model_lon.fit(
        X_train,
        y_lon_train,
    )


    MODEL_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )


    model_lat.save_model(
        MODEL_PATH / "latitude.cbm"
    )


    model_lon.save_model(
        MODEL_PATH / "longitude.cbm"
    )


    print(
        "Модели сохранены"
    )


    print(
        "\nВажность признаков:"
    )

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance":
                model_lat.feature_importances_,
        }
    )


    print(
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
