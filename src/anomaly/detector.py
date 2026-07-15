import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = {
    "speed_knots": 35.0,
    "course_change_degrees": 70.0,
    "calculated_speed_kmh": 100.0,
    "time_gap_minutes": 60.0,
}


def detect_rule_based_anomalies(
    data: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    result = data.copy()

    limits = DEFAULT_THRESHOLDS.copy()

    if thresholds:
        limits.update(thresholds)

    result["speed_anomaly"] = False
    result["course_anomaly"] = False
    result["position_jump_anomaly"] = False
    result["time_gap_anomaly"] = False

    if "sog" in result.columns:
        result["speed_anomaly"] = (
            result["sog"]
            > limits["speed_knots"]
        ).fillna(False)

    if "course_change" in result.columns:
        result["course_anomaly"] = (
            result["course_change"]
            > limits["course_change_degrees"]
        ).fillna(False)

    if "calculated_speed_kmh" in result.columns:
        result["position_jump_anomaly"] = (
            result["calculated_speed_kmh"]
            > limits["calculated_speed_kmh"]
        ).fillna(False)

    if "time_delta_seconds" in result.columns:
        result["time_gap_anomaly"] = (
            result["time_delta_seconds"]
            > limits["time_gap_minutes"] * 60
        ).fillna(False)

    anomaly_columns = [
        "speed_anomaly",
        "course_anomaly",
        "position_jump_anomaly",
        "time_gap_anomaly",
    ]

    result["anomaly_count"] = result[
        anomaly_columns
    ].sum(axis=1)

    result["is_anomaly"] = (
        result["anomaly_count"] > 0
    )

    result["risk_score"] = np.clip(
        result["anomaly_count"] * 25,
        0,
        100,
    )

    result["anomaly_reason"] = result.apply(
        build_anomaly_reason,
        axis=1,
    )

    return result


def build_anomaly_reason(row: pd.Series) -> str:
    reasons = []

    if row.get("speed_anomaly", False):
        reasons.append("необычно высокая скорость")

    if row.get("course_anomaly", False):
        reasons.append("резкое изменение курса")

    if row.get("position_jump_anomaly", False):
        reasons.append("скачок координат")

    if row.get("time_gap_anomaly", False):
        reasons.append("большой временной разрыв")

    if not reasons:
        return "нормальное поведение"

    return "; ".join(reasons)
