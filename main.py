# main.py
import argparse
import logging
from src.data import DataLoader

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

    # Здесь можно добавить другие команды, например:
    # ui_parser = subparsers.add_parser("ui", help="Запуск пользовательского интерфейса")

    args = parser.parse_args()

    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.command == "load":
        logger.info("Запуск загрузки и обработки данных...")
        loader = DataLoader(raw_data_path=args.raw_data_path)
        loader.process_all_data()
        logger.info("Обработка завершена.")

    # elif args.command == "ui":
    #     from src.UI.USER_INTERFACE_FIXED import main as ui_main
    #     ui_main()

if __name__ == "__main__":
    main()
