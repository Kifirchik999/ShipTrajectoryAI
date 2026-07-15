from pathlib import Path
import sys

import numpy as np
import pandas as pd


data_directory = Path("data/raw")


def get_size(value):
    if isinstance(value, np.ndarray):
        return f"shape={value.shape}, ndim={value.ndim}, dtype={value.dtype}"

    if isinstance(value, pd.Series):
        return f"length={len(value)}, dtype={value.dtype}"

    if isinstance(value, pd.DataFrame):
        return f"shape={value.shape}"

    try:
        return f"length={len(value)}"
    except TypeError:
        return "scalar"


def get_memory(value):
    if isinstance(value, np.ndarray):
        return value.nbytes

    if isinstance(value, pd.Series):
        return int(value.memory_usage(deep=True))

    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(deep=True).sum())

    return sys.getsizeof(value)


def inspect_file(file_path):
    print("=" * 90)
    print(f"Файл: {file_path.name}")
    print(f"Размер файла: {file_path.stat().st_size / 1024 / 1024:.2f} МБ")

    data = pd.read_pickle(file_path)

    print(f"Тип корневого объекта: {type(data)}")

    if isinstance(data, dict):
        print(f"Количество ключей: {len(data)}")
        print()

        total_memory = 0

        for key, value in data.items():
            memory = get_memory(value)
            total_memory += memory

            print(
                f"{key:20} "
                f"type={type(value).__name__:20} "
                f"{get_size(value):40} "
                f"memory={memory / 1024 / 1024:.4f} МБ"
            )

            if isinstance(value, np.ndarray):
                print(f"  первые значения: {value.reshape(-1)[:5]}")
            elif isinstance(value, pd.Series):
                print(f"  первые значения: {value.head().tolist()}")
            elif isinstance(value, (list, tuple)):
                print(f"  тип первого элемента: {type(value[0]) if value else None}")
                print(f"  первый элемент: {str(value[0])[:300]}")
            else:
                print(f"  значение: {str(value)[:300]}")

        print()
        print(
            "Суммарный размер значений в памяти: "
            f"{total_memory / 1024 / 1024:.2f} МБ"
        )

        try:
            frame = pd.DataFrame(data)

            print()
            print(f"DataFrame shape: {frame.shape}")
            print(frame.head(20).to_string())

        except Exception as error:
            print()
            print("Не удалось напрямую создать DataFrame:")
            print(error)

    elif isinstance(data, pd.DataFrame):
        print(f"DataFrame shape: {data.shape}")
        print(data.head())

    elif isinstance(data, (list, tuple)):
        print(f"Количество элементов: {len(data)}")

        if data:
            print(f"Тип первого элемента: {type(data[0])}")
            print(str(data[0])[:2000])


def main():
    files = list(data_directory.glob("*.pkl"))

    if not files:
        print("PKL-файлы не найдены")
        return

    for file_path in files:
        inspect_file(file_path)


if __name__ == "__main__":
    main()
