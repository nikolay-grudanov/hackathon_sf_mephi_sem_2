import pandas as pd
import os

# Пути к данным
RAW_DIR = "data/raw/"
INTERIM_DIR = "data/interim/"

# Список CSV-файлов
csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv") and not f.startswith("combined")]

# Вывод списка файлов
print("Найдены следующие CSV-файлы:")
for file in csv_files:
    print(f"- {file}")

# Список для датафреймов
dataframes = []

# Обработка каждого файла
for file in csv_files:
    try:
        df = pd.read_csv(RAW_DIR+file, encoding='utf-8')
        row_count = len(df)
        dataframes.append(df)
        print(f"Файл {os.path.basename(file)} содержит {row_count} строк")
    except Exception as e:
        print(f"Ошибка при обработке {file}: {str(e)}")

# Объединение датафреймов
combined_df = pd.concat(dataframes, ignore_index=True)

# Вывод итоговой информации
total_rows = len(combined_df)
print(f"\nИтоговый объединенный файл содержит {total_rows} строк")

# Сохранение результата
output_path = os.path.join(INTERIM_DIR, 'combined.csv')
combined_df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\nФайл сохранен: {output_path}")
