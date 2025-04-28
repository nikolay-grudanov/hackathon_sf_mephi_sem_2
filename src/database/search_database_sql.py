import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from sqlalchemy import create_engine, exc, inspect
from sqlalchemy.orm import sessionmaker, scoped_session
import logging

logger = logging.getLogger(__name__)
# Импорт базового класса и моделей
from src.database.models import Base, Document, Token 
class SearchDatabase:
    def __init__(self, db_path: str = None):
        self._init_db_path(db_path)
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
        logger.debug(f"База данных успешно создана: {self.db_path}")
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        self._create_tables()

    def _init_db_path(self, db_path: str):
        """Автоматическое определение пути к БД"""
        try:
            current_file = Path(__file__).resolve()
            root_dir = current_file.parent.parent if "src" in current_file.parts else current_file.parent
        except NameError:
            root_dir = Path(os.getcwd()).resolve()
        
        self.db_path = Path(db_path) if db_path else root_dir / "data/db/search.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _create_tables(self):
        try:
            Base.metadata.create_all(self.engine)
            logger.debug(f"Таблицы успешно созданы: {self.engine.table_names()}")
        except Exception as e:
            logger.error(f"Ошибка создания таблиц: {str(e)}")
            raise RuntimeError(f"Ошибка создания таблиц: {str(e)}")

    def add_document(self, doc_data: Dict) -> int:
        """
        Добавление документа с токенами
        :param doc_data: {
            'original_text': str,
            'cleaned_text': str,
            'tokens_data': List[str],
            'mapping': Dict[int, int],
            'embedding': bytes,
            'tokens': List[Dict] (поля Token)
        }
        """
        session = self.Session()
        try:
            doc = Document(
                original_text=doc_data['original_text'],
                cleaned_text=doc_data['cleaned_text'],
                tokens_data=json.dumps(doc_data['tokens_data']),
                mapping=json.dumps(doc_data['mapping']),
                embedding=doc_data['embedding']
            )
            session.add(doc)
            session.flush()

            for token_data in doc_data.get('tokens', []):
                token = Token(
                    document_id=doc.id,
                    **token_data
                )
                session.add(token)
            
            session.commit()
            return doc.id
        except exc.SQLAlchemyError as e:
            session.rollback()
            raise RuntimeError(f"Database error: {str(e)}")
        finally:
            session.close()

    def get_document(self, doc_id: int) -> Optional[Dict]:
        """Получение документа по ID"""
        session = self.Session()
        try:
            doc = session.query(Document).get(doc_id)
            if not doc:
                return None
            
            return {
                'id': doc.id,
                'original_text': doc.original_text,
                'cleaned_text': doc.cleaned_text,
                'tokens_data': json.loads(doc.tokens_data),
                'mapping': json.loads(doc.mapping),
                'embedding': doc.embedding,
                'tokens': [
                    {
                        'token': t.token,
                        'lemma': t.lemma,
                        'pos': t.pos,
                        'start_clean': t.start_clean,
                        'end_clean': t.end_clean,
                        'start_orig': t.start_orig,
                        'end_orig': t.end_orig
                    } for t in doc.tokens
                ]
            }
        except exc.SQLAlchemyError as e:
            raise RuntimeError(f"Database error: {str(e)}")
        finally:
            session.close()

    def __del__(self):
        self.Session.remove()