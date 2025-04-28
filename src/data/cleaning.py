import pandas as pd
from pathlib import Path
from src.utils.text_preprocessor import TextPreprocessor  # Используем существующий класс

# Корректный путь через относительные директории
def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent  # src/data/ → корень проекта

ROOT_DIR = get_project_root()
INTERIM_DIR = ROOT_DIR / "data" / "interim"

def clean_text(df):
    preprocessor = TextPreprocessor()
    for col in ["doc_text", "image2text", "speech2text"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: preprocessor.clear_text(x) if pd.notnull(x) else x)
    return df

# Загрузка объединенного файла
combined_path = INTERIM_DIR / "combined.csv"
if not combined_path.exists():
    raise FileNotFoundError(f"Файл {combined_path} не найден!")

combined_df = pd.read_csv(combined_path)
cleaned_df = clean_text(combined_df)

# Сохранение
cleaned_df.to_csv(INTERIM_DIR / "combined_cleaned.csv", index=False)