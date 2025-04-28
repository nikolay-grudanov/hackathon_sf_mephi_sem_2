import os
import logging
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class FaissDatabase:
    """
    Класс для работы с индексом Faiss и маппингом ID документов.
    Поддерживает разные типы индексов и гибкие настройки.
    """
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        index_dimension: int = 1024,
        index_type: str = "FlatIP",
        metric_type: str = "inner_product",
        overwrite: bool = False
    ):
        """
        Инициализация базы данных Faiss.
        
        Args:
            index_path (str): Путь к файлу индекса
            index_dimension (int): Размерность векторов
            index_type (str): Тип индекса ('FlatIP', 'FlatL2')
            metric_type (str): Тип метрики ('inner_product' или 'l2')
            overwrite (bool): Перезаписывать существующий индекс
        """
        self.index_dimension = index_dimension
        self.index_type = index_type
        self.metric_type = metric_type
        self.overwrite = overwrite
        
        self._init_paths(index_path)
        self._init_index()
        
        self.id_to_index: Dict[int, int] = {}
        self.index_to_id: Dict[int, int] = {}
        
        self._load_index(overwrite)

    def _init_paths(self, index_path: Optional[str]) -> None:
        """Инициализация путей к данным"""
        self.root_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.root_dir / "data/db/faiss_index"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if index_path:
            self.index_path = Path(index_path)
        else:
            self.index_path = self.data_dir / f"faiss_{self.index_type}.index"
        self.id_map_path = self.index_path.with_suffix('.pkl')
        
        self.tmp_index_path = self.index_path.with_suffix('.tmp')

    def _init_index(self) -> None:
        """Инициализация Faiss индекса"""
        if self.index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(self.index_dimension)
        elif self.index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(self.index_dimension)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def _load_index(self, overwrite: bool) -> None:
        """Загрузка или создание индекса"""
        if not overwrite and self.index_path.exists() and self.id_map_path.exists():
            self._load_from_disk()
        else:
            self._create_new_index()
            logger.info(f"Initialized new {self.index_type} index with dimension {self.index_dimension}")

    def _load_from_disk(self) -> None:
        """Загрузка индекса из файла"""
        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.id_map_path, "rb") as f:
                self.id_to_index = pickle.load(f)
            self.index_to_id = {v: k for k, v in self.id_to_index.items()}
            logger.info(f"Loaded Faiss index with {self.index.ntotal} embeddings")
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise

    def _create_new_index(self) -> None:
        """Создание нового индекса"""
        self._init_index()
        self.id_to_index = {}
        self.index_to_id = {}
        self._save_index()

    def close(self) -> None:
        """Сохранение индекса перед закрытием"""
        self._save_index()
        logger.info("Faiss index saved and closed")

    def _save_index(self) -> None:
        """Сохранение индекса с атомарной операцией"""
        if not self.index:
            raise ValueError("Index not initialized")
            
        try:
            # Сохранение во временные файлы
            faiss.write_index(self.index, str(self.tmp_index_path))
            with open(self.tmp_index_path.with_suffix('.pkl.tmp'), 'wb') as f:
                pickle.dump(self.id_to_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Атомарная замена файлов
            self.tmp_index_path.replace(self.index_path)
            self.tmp_index_path.with_suffix('.pkl.tmp').replace(self.id_map_path)
            logger.debug("Index and mappings successfully saved")
        except Exception as e:
            logger.error(f"Save failed: {str(e)}. Rolling back...")
            if self.tmp_index_path.exists():
                self.tmp_index_path.unlink()
            raise

    def add_embedding(self, embedding: np.ndarray, doc_id: int) -> None:
        """
        Добавление вектора в индекс с проверкой корректности
        
        Args:
            embedding (np.ndarray): Вектор размерности (1, D) или (D,)
            doc_id (int): Уникальный идентификатор документа
        """
        # Проверка размерности
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        if embedding.shape[1] != self.index_dimension:
            raise ValueError(f"Embedding must have {self.index_dimension} dimensions")
        
        # Проверка на дубли
        if doc_id in self.id_to_index:
            raise ValueError(f"Document ID {doc_id} already exists")
        
        # Нормализация вектора
        embedding = embedding.astype('float32')
        faiss.normalize_L2(embedding)
        
        # Добавление в индекс
        self.index.add(embedding)
        idx = self.index.ntotal - 1
        self.id_to_index[doc_id] = idx
        self.index_to_id[idx] = doc_id
        
        logger.debug(f"Added doc {doc_id} at index {idx}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Поиск похожих векторов
        
        Args:
            query_vector (np.ndarray): Запрос вектор (1, D) или (D,)
            top_k (int): Количество результатов
            threshold (float): Минимальный порог схожести
        
        Returns:
            Список словарей с doc_id, score и position
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape[1] != self.index_dimension:
            raise ValueError("Query vector has wrong dimension")
        
        # Нормализация и поиск
        query_vector = query_vector.astype('float32')
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, top_k)
        
        # Фильтрация результатов
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1 or idx not in self.index_to_id:
                continue
            if score < threshold:
                continue
            doc_id = self.index_to_id[idx]
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "position": idx
            })
        
        return results[:top_k]

    def delete_by_id(self, doc_id: int) -> bool:
        """Удаление вектора по doc_id"""
        if doc_id not in self.id_to_index:
            return False
        
        idx = self.id_to_index.pop(doc_id)
        del self.index_to_id[idx]
        
        # Обновление индекса (неэффективно для больших индексов)
        # Для больших данных лучше использовать динамические индексы
        new_vectors = []
        for i in range(self.index.ntotal):
            if i != idx:
                new_vectors.append(self.index.reconstruct(i))
        self.index = faiss.IndexFlatIP(self.index_dimension)
        self.index.add(np.stack(new_vectors))
        
        logger.debug(f"Removed doc {doc_id}")
        return True

    @property
    def size(self) -> int:
        """Общее количество векторов в индексе"""
        return self.index.ntotal if self.index else 0

    def __repr__(self) -> str:
        return (
            f"FaissDatabase("
            f"index_type={self.index_type}, "
            f"dimension={self.index_dimension}, "
            f"size={self.size})"
        )