import re

class TextPreprocessor:
    """
    Класс для очистки и предобработки текста.
    """

    __slots__ = (
        "url_pattern",
        "emoji_pattern",
        "html_pattern",
        "non_letter_pattern",
        "telegram_pattern",
        "spaces_pattern",
    )

    def __init__(self):
        """
        Инициализация класса TextPreprocessor.
        Предкомпилирует регулярные выражения для очистки текста.
        """
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # эмоции
            "\U0001f300-\U0001f5ff"  # символы
            "\U0001f680-\U0001f6ff"  # транспорт
            "\U0001f700-\U0001f77f"  # алхимия
            "\U0001f780-\U0001f7ff"  # геометрические фигуры
            "\U0001f800-\U0001f8ff"  # дополнительные символы
            "\U0001f900-\U0001f9ff"  # дополнительные символы-2
            "\U0001fa00-\U0001fa6f"  # шахматы
            "\U0001fa70-\U0001faff"  # дополнительные символы-3
            "\U00002702-\U000027b0"  # Dingbats
            "\U000024c2-\U0001f251"  # Enclosed
            "]+",
            flags=re.UNICODE,
        )
        self.html_pattern = re.compile(r"&[a-z]+;")
        self.non_letter_pattern = re.compile(r"[^a-zа-я\s]")
        self.telegram_pattern = re.compile(r"@\w+|/\w+")
        self.spaces_pattern = re.compile(r"\s+")

    def __preprocess(
        self,
        text,
        lowercase=True,
        replace_yo=True,
        remove_urls=True,
        remove_emoji=True,
        remove_html=True,
        remove_punctuation=True,
        remove_telegram=True,
    ):
        """
        Приватный метод базовой предобработки текста.

        :param text: Исходный текст для предобработки.
        :param lowercase: Флаг для приведения текста к нижнему регистру.
        :param replace_yo: Флаг для замены буквы "ё" на "е".
        :param remove_urls: Флаг для удаления URL-ссылок.
        :param remove_emoji: Флаг для удаления эмодзи.
        :param remove_html: Флаг для удаления HTML-сущностей.
        :param remove_punctuation: Флаг для удаления пунктуации и не-буквенных символов.
        :param remove_telegram: Флаг для удаления телеграм-упоминаний и бот-команд.
        :return: Очищенный текст.
        """
        result = text

        if lowercase:
            result = result.lower()

        if replace_yo:
            result = result.replace("ё", "е")

        if remove_urls:
            result = self.url_pattern.sub(" ", result)

        if remove_emoji:
            result = self.emoji_pattern.sub(" ", result)

        if remove_html:
            result = self.html_pattern.sub(" ", result)

        if remove_punctuation:
            result = self.non_letter_pattern.sub(" ", result)

        if remove_telegram:
            result = self.telegram_pattern.sub(" ", result)

        return self.spaces_pattern.sub(" ", result).strip()

    def clear_text(self, text):
        """
        Полная очистка текста для семантического поиска.

        :param text: Исходный текст для очистки.
        :return: Очищенный текст.
        """
        return self.__preprocess(text)

    def clean_for_search(self, text):
        """
        Очистка текста для поисковых запросов.

        :param text: Исходный текст для очистки.
        :return: Очищенный текст.
        """
        return self.__preprocess(text, remove_punctuation=False, remove_telegram=False)

    def clean_for_embedding(self, text):
        """
        Очистка текста перед получением эмбеддингов.

        :param text: Исходный текст для очистки.
        :return: Очищенный текст.
        """
        return self.__preprocess(
            text, remove_emoji=True, remove_html=True, remove_punctuation=True
        )

    def clean_for_display(self, text):
        """
        Лёгкая очистка для отображения пользователю.

        :param text: Исходный текст для очистки.
        :return: Очищенный текст.
        """
        return self.__preprocess(
            text,
            lowercase=False,
            replace_yo=False,
            remove_urls=False,
            remove_emoji=False,
            remove_punctuation=False,
        )

    def get_text_stats(self, text):
        """
        Возвращает статистику по тексту: длина текста, количество слов,
        уникальные слова и другие полезные метрики.

        Args:
            text (str): Исходный текст для анализа

        Returns:
            dict: Словарь с различными статистическими метриками текста:
                - length: общая длина текста в символах
                - word_count: количество слов в тексте
                - unique_word_count: количество уникальных слов
                - word_frequencies: словарь частоты встречаемости слов
                - avg_word_length: средняя длина слова
                - short_words_count: количество слов длиной менее 4 символов
                - long_words_count: количество слов длиной более 7 символов
        """
        # Проверяем входные данные
        if not text or not isinstance(text, str):
            return {
                "length": 0,
                "word_count": 0,
                "unique_word_count": 0,
                "word_frequencies": {},
            }

        # Очищаем текст 
        cleaned_text = self.clear_text(text)

        # Разбиваем текст на слова
        words = cleaned_text.split()

        # Рассчитываем основные метрики
        word_count = len(words)
        unique_words = set(words)
        unique_word_count = len(unique_words)

        # Рассчитываем частоту слов
        word_frequencies = {}
        for word in words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

        # Рассчитываем дополнительные метрики
        total_chars = sum(len(word) for word in words)
        avg_word_length = total_chars / word_count if word_count > 0 else 0
        short_words_count = sum(1 for word in words if len(word) < 4)
        long_words_count = sum(1 for word in words if len(word) > 7)

        # Собираем все метрики в словарь
        stats = {
            "length": len(cleaned_text),  # Длина текста в символах
            "word_count": word_count,  # Количество слов
            "unique_word_count": unique_word_count,  # Количество уникальных слов
            "word_frequencies": word_frequencies,  # Частота слов
            "avg_word_length": round(avg_word_length, 2),  # Средняя длина слова
            "short_words_count": short_words_count,  # Количество коротких слов (< 4 букв)
            "long_words_count": long_words_count,  # Количество длинных слов (> 7 букв)
        }

        return stats