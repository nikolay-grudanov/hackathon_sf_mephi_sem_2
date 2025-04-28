from src.database.faiss_database import FaissDatabase
from src.utils.natasha_linguistic_analyzer import NatashaLinguisticAnalyzer
from src.utils.sbert_large_nlu import SbertLargeNLU

class SearchEngine:
    def __init__(self):
        self.natasha = NatashaLinguisticAnalyzer()
        self.sbert = SbertLargeNLU(enable_rocm=True)
        self.faiss_db = FaissDatabase()

    def search(self, query, top_k=5):
        # Предобработка запроса Natasha
        analysis = self.natasha.analyze_query(query)
        cleaned_query = analysis['cleaned_text']
        
        # Генерация эмбеддинга
        query_emb = self.sbert.create_embeddings(cleaned_query)
        
        # Поиск в FAISS
        results = self.faiss_db.search(query_emb, top_k)
        
        # Возвращаем doc_id и scores
        return results