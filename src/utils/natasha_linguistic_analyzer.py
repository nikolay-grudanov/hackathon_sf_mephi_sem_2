import re
from typing import List, Tuple, Dict, Optional

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)

from .text_preprocessor import TextPreprocessor

import logging

logger = logging.getLogger(__name__)

class NatashaLinguisticAnalyzer:
    """
    Класс для лингвистического анализа русского текста с использованием Natasha.
    
    Выполняет токенизацию, лемматизацию, определение частей речи и
    сохраняет позиции символов каждого токена в оригинальном тексте.
    """
    
    __slots__ = ('segmenter', 'morph_vocab', 'emb', 'morph_tagger', 'text_preprocessor')
    
    def __init__(self):
        """
        Инициализирует компоненты Natasha для лингвистического анализа.
        
        Используемые компоненты:
        - Segmenter: Для сегментации текста на токены
        - MorphVocab: Для морфологического словаря и лемматизации
        - NewsEmbedding: Для векторных представлений
        - NewsMorphTagger: Для морфологической разметки (определения частей речи)
        - TextPreprocessor: Для предобработки текста
        """
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)

        # Инициализация экземпляра TextPreprocessor (композиция)
        self.text_preprocessor = TextPreprocessor()
        logger.debug('Наташа инициализирована.')
    
    def clear_text_with_mapping(self, text: str) -> Tuple[str, List[int]]:
        """
        Очищает текст и возвращает маппинг индексов из очищенного текста в исходный.
        """
        # Получаем очищенный текст с помощью TextPreprocessor
        cleaned_text = self.text_preprocessor.clear_text(text)
        logger.debug(f'Оригинальный текст: {text}')
        logger.debug(f'Очищенный текст: {cleaned_text}')
        
        # Создаем маппинг индексов из очищенного текста в исходный
        mapping = []
        orig_index = 0
        
        for clean_char in cleaned_text:
            found = False
            # Пропускаем символы в исходном тексте до нахождения соответствия
            while orig_index < len(text) and not found:
                if text[orig_index].lower() == clean_char:
                    mapping.append(orig_index)
                    orig_index += 1
                    found = True
                else:
                    orig_index += 1
            
            # Если не найдено соответствие, используем последний известный индекс
            if not found:
                # Добавляем последний возможный индекс или -1, если текст пустой
                mapping.append(len(text) - 1 if text else -1)
        
        return cleaned_text, mapping

    
    def map_clean_to_original(self, clean_start: int, clean_end: int, mapping: List[int]) -> Tuple[int, int]:
        """
        Преобразует позицию в очищенном тексте в позицию в исходном тексте.
        """
        if not mapping:
            return -1, -1
        
        # Проверка границ
        if clean_start >= len(mapping):
            return -1, -1
        
        orig_start = mapping[clean_start]
        
        # Обработка случая, когда clean_end выходит за границы
        if clean_end - 1 >= len(mapping):
            # Используем последний элемент маппинга + некоторое смещение, 
            # чтобы захватить позиции после последнего маппированного символа
            orig_end = mapping[-1] + 2  # +2 для захвата большего контекста
        else:
            orig_end = mapping[clean_end - 1] + 1
        
        return orig_start, orig_end

    
    def analyze(self, text: str) -> Dict:
        """
        Выполняет лингвистический анализ текста с использованием Natasha.
        
        Args:
            text (str): Входной текст для анализа.
            
        Returns:
            Dict: Словарь с результатами анализа:
                - 'tokens': Список кортежей с информацией о токенах
                  (token, lemma, pos_tag, start_clean, end_clean, start_orig, end_orig)
                - 'original_text': Оригинальный текст (до очистки)
                - 'cleaned_text': Очищенный текст (после предобработки)
                - 'mapping': Маппинг индексов из очищенного текста в исходный
        """
        if not text or not isinstance(text, str):
            return {'tokens': [], 'original_text': '', 'cleaned_text': '', 'mapping': []}
        
        # Сохраняем оригинальный текст
        original_text = text
        
        # Базовая предобработка с TextPreprocessor и получение маппинга
        cleaned_text, mapping = self.clear_text_with_mapping(text)

        # Создаем объект Doc из Natasha
        doc = Doc(cleaned_text)
        
        # Сегментация текста на токены
        doc.segment(self.segmenter)
        
        # Морфологический анализ (определение частей речи)
        doc.tag_morph(self.morph_tagger)
        
        # Лемматизация токенов
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        
        # Создаем результирующий список с позициями как в очищенном, так и в исходном тексте
        tokens = []
        for token in doc.tokens:
            # Позиции в очищенном тексте
            start_clean = token.start
            end_clean = token.stop
            
            # Маппинг в позиции в исходном тексте
            start_orig, end_orig = self.map_clean_to_original(start_clean, end_clean, mapping)
            
            tokens.append((
                token.text,          # Оригинальный токен
                token.lemma,         # Лемматизированная форма
                token.pos,           # Часть речи
                start_clean,         # Начальный индекс в очищенном тексте
                end_clean,           # Конечный индекс в очищенном тексте
                start_orig,          # Начальный индекс в исходном тексте
                end_orig             # Конечный индекс в исходном тексте
            ))
        logger.debug(f'Токены после анализа: {tokens}')

        return {
            'tokens': tokens,
            'original_text': original_text,
            'cleaned_text': cleaned_text,
            'mapping': mapping
        }
    
    def analyze_query(self, text: str) -> Dict:
        """
        Анализирует поисковый запрос и возвращает информацию о токенах и текстах.
        
        Args:
            text (str): Исходный текст поискового запроса
            
        Returns:
            Dict: Словарь с результатами анализа (аналогично методу analyze)
        """
        return self.analyze(text)