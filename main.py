from src.utils.sbert_large_nlu import SbertLargeNLU
from src.database import SearchDatabase
from sqlalchemy import inspect

from src.database.faiss_database import FaissDatabase
from src.utils.sbert_large_nlu import SbertLargeNLU

# Инициализация

# Инициализация модели
model = SbertLargeNLU(enable_rocm=True)
print(model.device)  # Должно вывести: cuda
embeddings = model.create_embeddings("Тестовый текст")
print(embeddings.shape)  # Ожидается: torch.Size([1, 1024])

# Инициализация базы данных
db = SearchDatabase()
inspector = inspect(db.engine)
print(f"Таблицы созданы: {inspector.get_table_names()}")

# Инициализация базы данных с помощью Faiss
faiss_db = FaissDatabase()

# Генерация эмбеддингов
texts = ["Пример текста 1", "Пример текста 2"]
embeddings = model.create_embeddings(texts)

# Сохранение
for idx, emb in enumerate(embeddings):
    faiss_db.save_embeddings(emb.reshape(1, -1), doc_id=idx+1)

# Поиск
query_embedding = model.create_embeddings(["Пример текста 1"])
results = faiss_db.search(query_embedding, top_k=2)
print(results)  # [{doc_id: 1, score: 0.95}, ...]




