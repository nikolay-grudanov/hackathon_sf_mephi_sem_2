from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Явный импорт всех моделей
from .document import Document
from .token import Token

__all__ = ['Base', 'Document', 'Token']
