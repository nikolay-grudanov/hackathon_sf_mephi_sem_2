import numpy as np
from src.utils.sbert_large_nlu import SbertLargeNLU
from src.database.faiss_database import FaissDatabase
from src.database.search_database_sql import SearchDatabase

# Инициализация модели
model = SbertLargeNLU(enable_rocm=True)
print(model.device)  
embeddings = model.create_embeddings("Тестовый текст")
print(embeddings.shape)  

# Инициализация баз данных
sql_db = SearchDatabase()
faiss_db = FaissDatabase()  # Инициализация с параметрами по умолчанию

# Генерация эмбеддингов и сохранение
texts = ["Пример текста 1", "Пример текста 2"]
embeddings = model.create_embeddings(texts)

for idx, emb in enumerate(embeddings):
    # Добавляем batch-размерность (1, 1024) и конвертируем в float32
    faiss_db.add_embedding(
        np.expand_dims(emb, axis=0).astype('float32'),
        doc_id=idx + 1
    )

# Поиск
query = "Пример текста 1"
query_embedding = model.create_embeddings([query])  # Обратите внимание на список
query_embedding = np.expand_dims(query_embedding[0], axis=0).astype('float32')  # Берем первый элемент
results = faiss_db.search(query_embedding, top_k=2)
print(results)  

