import argparse
import logging
import os
from src.data import DataLoader
from src.search import SearchEngine
from src.database.search_database_sql import SearchDatabase

def check_db_exists():
    db = SearchDatabase()
    db_path = db.db_path
    faiss_index_path = db_path.parent / "faiss_index" / "faiss_FlatIP.index"
    if not db_path.exists():
        print(f"База данных не найдена: {db_path}")
        return False
    if not faiss_index_path.exists():
        print(f"FAISS-индекс не найден: {faiss_index_path}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="CLI для управления системой семантического поиска")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Доступные команды")

    # Команда загрузки данных
    load_parser = subparsers.add_parser("load", help="Загрузка, очистка и индексация данных")
    load_parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/raw",
        help="Путь к папке с исходными CSV-файлами (по умолчанию: data/raw)"
    )

    # Команда поиска
    search_parser = subparsers.add_parser("search", help="Семантический поиск по базе")
    search_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Строка поискового запроса"
    )
    search_parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Сколько результатов выводить (по умолчанию 5)"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.command == "load":
        logger.info("Запуск загрузки и обработки данных...")
        loader = DataLoader(raw_data_path=args.raw_data_path)
        loader.process_all_data()
        logger.info("Обработка завершена.")

    elif args.command == "search":
        if not check_db_exists():
            logger.error("База данных или индекс не найдены. Сначала выполните загрузку данных (load).")
            return
        engine = SearchEngine()  # <-- используем SearchEngine!
        logger.info(f"Выполняется поиск по запросу: {args.query!r}")
        results = engine.search(args.query, top_k=args.top_k)
        if not results:
            print("Совпадений не найдено.")
        else:
            print(f"Топ-{args.top_k} результатов:")
            for i, res in enumerate(results, 1):
                print(f"\n=== Результат #{i} ===")
                print(f"Документ ID: {res['doc_id']}")
                print(f"Счет (score): {res['score']:.3f}")
                # Здесь можно добавить вывод фрагмента, если SearchEngine будет возвращать его

if __name__ == "__main__":
    main()
