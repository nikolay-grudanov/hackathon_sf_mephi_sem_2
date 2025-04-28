from sqlalchemy import Column, Integer, Text, LargeBinary, Index
from sqlalchemy.orm import relationship
from src.database.models import Base 

class Document(Base):
    """Модель документа с текстовыми данными и эмбеддингами."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    original_text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=True)
    # Храним JSON-сериализованный список токенов
    tokens_data = Column(Text, nullable=True)
    # Храним JSON-сериализованный маппинг индексов
    mapping = Column(Text, nullable=True)
    # Храним сериализованный numpy array
    embedding = Column(LargeBinary, nullable=True)
    
    # Определяем отношение один-ко-многим с токенами
    tokens = relationship("Token", back_populates="document", cascade="all, delete-orphan")
    
    # Индексы для быстрого поиска
    __table_args__ = (
        Index('idx_documents_original_text', original_text),
        Index('idx_documents_cleaned_text', cleaned_text),
    )