from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COMPARISON_PATH = Path("reports/tables/model_comparison.csv")
IMPORTANCE_PATH = Path(
    "reports/tables/catboost_feature_importance.csv"
)

FIGURES_DIRECTORY = Path("reports/figures")


def create_comparison_chart(data):
    table = data.pivot(
        index="horizon_steps",
        columns="model",
        values="mean_error_km",
    )

    table = table * 1000

    figure, axis = plt.subplots(figsize=(9, 5))

    table.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "Сравнение точности прогнозирования"
    )

    axis.set_xlabel(
        "Горизонт прогнозирования, AIS-шаги"
    )

    axis.set_ylabel(
        "Средняя ошибка, метры"
    )

    axis.tick_params(
        axis="x",
        rotation=0,
    )

    axis.legend(
        title="Модель"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIRECTORY
        / "model_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_improvement_chart(data):
    baseline = data[
        data["model"] == "Constant Velocity"
    ].set_index("horizon_steps")

    catboost = data[
        data["model"] == "CatBoost"
    ].set_index("horizon_steps")

    improvement = (
        (
            baseline["mean_error_km"]
            - catboost["mean_error_km"]
        )
        / baseline["mean_error_km"]
        * 100
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    improvement.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "Улучшение CatBoost относительно baseline"
    )

    axis.set_xlabel(
        "Горизонт прогнозирования, AIS-шаги"
    )

    axis.set_ylabel(
        "Снижение средней ошибки, %"
    )

    axis.tick_params(
        axis="x",
        rotation=0,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIRECTORY
        / "catboost_improvement.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_feature_importance_chart(data):
    importance = (
        data.groupby("feature")["importance"]
        .mean()
        .sort_values()
    )

    figure, axis = plt.subplots(figsize=(9, 6))

    importance.plot(
        kind="barh",
        ax=axis,
    )

    axis.set_title(
        "Средняя важность признаков CatBoost"
    )

    axis.set_xlabel(
        "Важность признака"
    )

    axis.set_ylabel(
        "Признак"
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIRECTORY
        / "catboost_feature_importance.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_summary_table(data):
    baseline = data[
        data["model"] == "Constant Velocity"
    ].copy()

    catboost = data[
        data["model"] == "CatBoost"
    ].copy()

    summary = baseline[
        [
            "horizon_steps",
            "mean_error_km",
            "rmse_km",
            "mean_ade_km",
            "mean_fde_km",
        ]
    ].merge(
        catboost[
            [
                "horizon_steps",
                "mean_error_km",
                "rmse_km",
                "mean_ade_km",
                "mean_fde_km",
            ]
        ],
        on="horizon_steps",
        suffixes=(
            "_baseline",
            "_catboost",
        ),
    )

    summary["improvement_percent"] = (
        (
            summary["mean_error_km_baseline"]
            - summary["mean_error_km_catboost"]
        )
        / summary["mean_error_km_baseline"]
        * 100
    )

    summary["baseline_error_m"] = (
        summary["mean_error_km_baseline"]
        * 1000
    )

    summary["catboost_error_m"] = (
        summary["mean_error_km_catboost"]
        * 1000
    )

    output_path = Path(
        "reports/tables/final_model_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return output_path


def main():
    FIGURES_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = pd.read_csv(
        COMPARISON_PATH
    )

    importance = pd.read_csv(
        IMPORTANCE_PATH
    )

    comparison_chart = create_comparison_chart(
        comparison
    )

    improvement_chart = create_improvement_chart(
        comparison
    )

    importance_chart = create_feature_importance_chart(
        importance
    )

    summary_table = create_summary_table(
        comparison
    )

    print("Готово")
    print(f"Сравнение моделей: {comparison_chart}")
    print(f"Улучшение CatBoost: {improvement_chart}")
    print(f"Важность признаков: {importance_chart}")
    print(f"Итоговая таблица: {summary_table}")


if __name__ == "__main__":
    main()
