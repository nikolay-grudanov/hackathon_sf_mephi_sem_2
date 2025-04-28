from src.utils.natasha_linguistic_analyzer import NatashaLinguisticAnalyzer
from src.utils.sbert_large_nlu import SbertLargeNLU
from src.database.faiss_database import FaissDatabase
from src.database.search_database_sql import SearchDatabase

class DataLoader:
    def __init__(self):
        self.natasha = NatashaLinguisticAnalyzer()
        self.sbert = SbertLargeNLU(enable_rocm=True)
        self.faiss_db = FaissDatabase()
        self.sql_db = SearchDatabase()

    def process_and_index(self, texts):
        for idx, text in enumerate(texts):
            # Предобработка Natasha
            analysis = self.natasha.analyze(text)
            cleaned_text = analysis['cleaned_text']
            
            # Генерация эмбеддингов
            embeddings = self.sbert.create_embeddings(cleaned_text)
            
            # Сохранение в SQL
            doc_id = self.sql_db.add_document({
                'original_text': text,
                'cleaned_text': cleaned_text,
                'tokens_data': analysis['tokens'],
                'embedding': embeddings.tobytes()
            })
            
            # Индексация в FAISS
            self.faiss_db.save_embeddings(embeddings.reshape(1, -1), doc_id=doc_id)