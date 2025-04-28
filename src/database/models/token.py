
from sqlalchemy import Column, Integer, String, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Создаем базовый класс для моделей SQLAlchemy
Base = declarative_base()

class Token(Base):
    """Модель токена с информацией о лемме, POS-теге и позиции."""
    __tablename__ = "tokens"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    token = Column(String(255), nullable=False)
    lemma = Column(String(255), nullable=True)
    pos = Column(String(50), nullable=True)
    start_clean = Column(Integer, nullable=True)
    end_clean = Column(Integer, nullable=True)
    start_orig = Column(Integer, nullable=True)
    end_orig = Column(Integer, nullable=True)
    
    # Определяем обратное отношение к документу
    document = relationship("Document", back_populates="tokens")
    
    # Индексы для быстрого поиска
    __table_args__ = (
        Index('idx_tokens_document_id', document_id),
        Index('idx_tokens_lemma', lemma),
        Index('idx_tokens_pos', pos),
    )