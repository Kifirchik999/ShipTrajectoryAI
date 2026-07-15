from pathlib import Path

import pandas as pd


DATA_DIRECTORY = Path("data/raw")


def inspect_pickle(file_path: Path) -> None:
    data = pd.read_pickle(file_path)

    print("=" * 80)
    print(f"Файл: {file_path.name}")
    print(f"Размер: {file_path.stat().st_size / 1024 / 1024:.2f} МБ")
    print(f"Тип объекта: {type(data)}")

    if not isinstance(data, dict):
        print("Файл содержит не словарь")
        print(str(data)[:2000])
        return

    print("\nСтруктура словаря:")

    for key, value in data.items():
        value_type = type(value).__name__

        try:
            value_length = len(value)
        except TypeError:
            value_length = "нет длины"

        print(
            f"{key:15} "
            f"тип={value_type:15} "
            f"длина={value_length}"
        )

    try:
        frame = pd.DataFrame(data)
    except Exception as error:
        print("\nНе удалось преобразовать словарь в таблицу:")
        print(error)
        return

    print("\nТаблица успешно создана")
    print(f"Количество строк: {len(frame)}")
    print(f"Количество столбцов: {len(frame.columns)}")

    print("\nСтолбцы:")
    print(frame.columns.tolist())

    print("\nПервые строки:")
    print(frame.head())

    print("\nТипы данных:")
    print(frame.dtypes)

    print("\nПропущенные значения:")
    print(frame.isna().sum())

    print("\nКоличество уникальных судов:")
    print(frame["mmsi"].nunique())

    print("\nКоличество уникальных типов судов:")
    print(frame["shiptype"].nunique())

    print("\nСтатистика длины траекторий:")
    print(frame["track_length"].describe())

    output_path = Path("data/processed/ais_initial.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_size = min(100000, len(frame))

    frame.head(sample_size).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nПервые {sample_size} строк сохранены в:"
        f"\n{output_path}"
    )


def main() -> None:
    pickle_files = list(DATA_DIRECTORY.glob("*.pkl"))

    if not pickle_files:
        print("В папке data/raw нет PKL-файлов")
        return

    for file_path in pickle_files:
        inspect_pickle(file_path)


if __name__ == "__main__":
    main()
