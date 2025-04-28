import os
import logging
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class FaissDatabase:
    def __init__(self, index_path: Optional[str] = None):
        self._init_paths(index_path)
        self.index: Optional[faiss.Index] = None
        self.id_to_index: Dict[int, int] = {}
        self.index_to_id: Dict[int, int] = {}
        self._load_or_create_index()

    def _init_paths(self, index_path: Optional[str]) -> None:
        """Инициализация путей с автоматическим определением корневой директории"""
        self.root_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.root_dir / "data/db/faiss_index"
        
        if index_path:
            self.index_path = Path(index_path)
        else:
            self.index_path = self.data_dir / "faiss_index.bin"
        
        self.id_map_path = self.index_path.with_suffix('.pkl')
        self.tmp_index_path = self.index_path.with_suffix('.tmp')
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_create_index(self) -> None:
        """Загрузка или создание индекса с обработкой ошибок"""
        try:
            if self.index_path.exists() and self.id_map_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                with open(self.id_map_path, "rb") as f:
                    self.id_to_index = pickle.load(f)
                self.index_to_id = {v: k for k, v in self.id_to_index.items()}
                logger.info("FAISS index loaded successfully")
            else:
                self._create_new_index()
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self._create_new_index()

    def _create_new_index(self) -> None:
        """Инициализация нового индекса с L2 нормализацией"""
        self.index = faiss.IndexFlatIP(1024)
        self.id_to_index = {}
        self.index_to_id = {}
        logger.info("Created new FAISS index")
        self._save_index()

    def __enter__(self):
        """Контекстный менеджер для безопасной работы"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Гарантированное сохранение при выходе из контекста"""
        self.close()

    def close(self) -> None:
        """Явное закрытие с обработкой ошибок"""
        try:
            self._save_index()
            logger.info("Index saved successfully on close")
        except Exception as e:
            logger.error(f"Error during index closing: {e}")
            raise

    def _save_index(self) -> None:
        """Атомарное сохранение индекса через временный файл"""
        if not self.index:
            return

        try:
            # Сохраняем во временный файл
            faiss.write_index(self.index, str(self.tmp_index_path))
            with open(self.tmp_index_path.with_suffix('.pkl.tmp'), 'wb') as f:
                pickle.dump(self.id_to_index, f, protocol=5)

            # Перемещаем временные файлы в целевые
            self.tmp_index_path.replace(self.index_path)
            self.tmp_index_path.with_suffix('.pkl.tmp').replace(self.id_map_path)
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            # Удаляем временные файлы при ошибке
            if self.tmp_index_path.exists():
                self.tmp_index_path.unlink()
            if self.tmp_index_path.with_suffix('.pkl.tmp').exists():
                self.tmp_index_path.with_suffix('.pkl.tmp').unlink()
            raise

    def save_embeddings(self, embedding: np.ndarray, doc_id: int) -> None:
        """Добавление эмбеддинга с проверкой дубликатов"""
        if not self.index:
            raise ValueError("Index not initialized")

        if doc_id in self.id_to_index:
            raise ValueError(f"Document ID {doc_id} already exists")

        # Нормализация и преобразование типа
        embedding = embedding.astype('float32')
        faiss.normalize_L2(embedding)

        # Добавление в индекс
        self.index.add(embedding)
        idx = self.index.ntotal - 1
        
        # Обновление маппинга
        self.id_to_index[doc_id] = idx
        self.index_to_id[idx] = doc_id
        logger.debug(f"Added embedding for doc_id {doc_id} at index {idx}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Семантический поиск с нормализацией запроса"""
        if not self.index or self.index.ntotal == 0:
            return []

        # Подготовка запроса
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)

        # Поиск и формирование результатов
        distances, indices = self.index.search(query_embedding, top_k)
        return [
            {
                "doc_id": self.index_to_id[idx],
                "score": float(score),
                "index_position": int(idx)
            }
            for idx, score in zip(indices[0], distances[0])
            if idx != -1 and idx in self.index_to_id
        ]

    @property
    def size(self) -> int:
        """Текущий размер индекса"""
        return self.index.ntotal if self.index else 0

    def __repr__(self) -> str:
        return f"FaissDatabase(index_path={self.index_path}, size={self.size})"