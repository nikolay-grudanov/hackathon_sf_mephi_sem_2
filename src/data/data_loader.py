from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import json
import logging

from src.utils.natasha_linguistic_analyzer import NatashaLinguisticAnalyzer
from src.utils.sbert_large_nlu import SbertLargeNLU
from src.database.faiss_database import FaissDatabase
from src.database.search_database_sql import SearchDatabase, Document, Token

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, raw_data_path: str = "data/raw"):
        self.natasha = NatashaLinguisticAnalyzer()
        self.sbert = SbertLargeNLU(enable_rocm=True)
        self.faiss_db = FaissDatabase()
        self.sql_db = SearchDatabase()
        self.raw_data_path = Path(raw_data_path)
        self.seen_texts = set()

    def _load_raw_data(self) -> pd.DataFrame:
        """Загрузка данных из CSV-файлов"""
        files = list(self.raw_data_path.glob("*.csv"))
        logger.info(f"Loading data from {len(files)} CSV files...")
        if not files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_data_path}")
        return pd.concat(
            [pd.read_csv(f, usecols=["doc_text"]) for f in files],
            ignore_index=True
        )

    def _preprocess_text(self, row: pd.Series) -> str:
        """Очистка только doc_text"""
        text = str(row["doc_text"]).strip()
        if text and text.lower() != "nan":
            cleaned = self.natasha.text_preprocessor.clear_text(text)
            return cleaned[:500]  # Ограничение длины
        return ""

    def _process_batch(self, batch: List[Dict], results: Dict) -> None:
        try:
            texts = [doc['cleaned_text'] for doc in batch]
            embeddings = self.sbert.create_embeddings(texts)

            doc_ids = []
            with self.sql_db.Session() as session:
                docs = []
                for i, doc_data in enumerate(batch):
                    doc = Document(
                        original_text=doc_data['original_text'],
                        cleaned_text=doc_data['cleaned_text'],
                        tokens_data=json.dumps(doc_data['tokens_data']),
                        mapping=json.dumps(doc_data['mapping']),
                        embedding=embeddings[i].tobytes()
                    )
                    session.add(doc)
                    session.flush()  # Получаем doc.id сразу после добавления
                    doc_ids.append(doc.id)
                    docs.append(doc)
                    # Добавляем токены для документа
                    for token_data in doc_data['tokens']:
                        token_obj = Token(document_id=doc.id, **token_data)
                        session.add(token_obj)
                session.commit()

            # Добавляем эмбеддинги в FAISS (уже вне сессии)
            for emb, doc_id in zip(embeddings, doc_ids):
                self.faiss_db.add_embedding(emb, doc_id)

            results['success'] += len(batch)
            results['doc_ids'].extend(doc_ids)

        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            results['errors'] += len(batch)
            results['error_messages'].append(str(e))

    def process_and_index(self, rows: List[dict], batch_size: int = 32) -> Dict[str, Any]:
        """Основной метод пакетной обработки"""
        results = {
            'success': 0,
            'errors': 0,
            'duplicates': 0,
            'doc_ids': [],
            'error_messages': []
        }

        current_batch = []

        for row in tqdm(rows, desc="Обработка документов"):
            try:
                original_text = str(row.get("doc_text", "")).strip()
                if not original_text:
                    continue

                # Лингвистический анализ по ОРИГИНАЛУ!
                analyzed = self.natasha.analyze(original_text)
                cleaned_text = analyzed['cleaned_text']

                # Проверка дубликатов по очищенному тексту
                text_hash = hash(cleaned_text.strip().lower())
                if text_hash in self.seen_texts:
                    results['duplicates'] += 1
                    continue
                self.seen_texts.add(text_hash)

                tokens = analyzed['tokens']
                if len(tokens) > 30:
                    tokens = tokens[:30]
                    analyzed['cleaned_text'] = ' '.join([t[0] for t in tokens])

                doc_data = {
                    'original_text': analyzed['original_text'],
                    'cleaned_text': analyzed['cleaned_text'],
                    'tokens_data': [token[:4] for token in tokens],
                    'mapping': analyzed['mapping'],
                    'tokens': [{
                        'token': token[0],
                        'lemma': token[1],
                        'pos': token[2],
                        'start_clean': token[3],
                        'end_clean': token[4],
                        'start_orig': token[5],
                        'end_orig': token[6]
                    } for token in tokens]
                }

                current_batch.append(doc_data)

                # Пакетная обработка
                if len(current_batch) >= batch_size:
                    self._process_batch(current_batch, results)
                    current_batch = []

            except Exception as e:
                results['errors'] += 1
                results['error_messages'].append(str(e))
                logger.error(f"Обработка текста завершилась ошибкой: {str(e)}")

        # Обработка последнего пакета
        if current_batch:
            self._process_batch(current_batch, results)

        return results

    def process_all_data(self) -> None:
        """Полный цикл обработки данных"""
        logger.debug("Загрузка и предобработка данных...")
        df = self._load_raw_data()
        logger.debug("Данные загружены и предобработаны.")
        df["processed_text"] = df.apply(self._preprocess_text, axis=1)

        # Передаем только doc_text и processed_text
        valid_rows = df[
            df["processed_text"].str.split().str.len().between(1, 30)
        ][["doc_text", "processed_text"]].to_dict(orient="records")

        results = self.process_and_index(valid_rows)

        self.faiss_db.save_index()
        logger.info(
            f"Обработка завершена. Успешно: {results['success']}, "
            f"Ошибки: {results['errors']}, Дубликаты: {results['duplicates']}"
        )