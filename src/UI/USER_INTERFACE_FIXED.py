import pandas as pd
import streamlit as st
import faiss
import numpy as np
import os
import pickle
import re
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import string

# Инициализация модели Sberbank
@st.cache_resource
def load_model():
    return SentenceTransformer('sberbank-ai/sbert_large_nlu_ru', device='cpu')

model = load_model()

# Конфигурация
VECTOR_FILE = 'social_vectors.pkl'
INDEX_FILE = 'social_faiss.index'
DATA_PATHS = [
    'D:/dzml/classes/1.csv',
    'D:/dzml/classes/2.csv',
    'D:/dzml/classes/3.csv',
    'D:/dzml/classes/4.csv',
    'D:/dzml/classes/5.csv',
    'D:/dzml/classes/6.csv'
]

# Функции предобработки текста
def clean_text(text):
    """Глубокая очистка текста из соцсетей"""
    if not isinstance(text, str):
        return ""
    
    # Удаление спецсимволов и ссылок
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    # Удаление эмодзи и лишних пробелов
    text = re.sub(r'[^\w\s.,!?а-яА-ЯёЁ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def preprocess_post(row):
    """Объединение и обработка всех текстовых полей поста"""
    texts = [
        clean_text(row.get('doc_text', '')),
        clean_text(row.get('image2text', '')),
        clean_text(row.get('speech2text', ''))
    ]
    return ' '.join([t for t in texts if t])

# Загрузка и подготовка данных
@st.cache_data
def load_and_preprocess():
    all_posts = []
    for path in DATA_PATHS:
        try:
            df = pd.read_csv(path)
            df['processed_text'] = df.apply(preprocess_post, axis=1)
            all_posts.extend(df[df['processed_text'].str.len() > 10]['processed_text'].tolist())
        except Exception as e:
            st.error(f"Ошибка загрузки {path}: {str(e)}")
    return all_posts

documents = load_and_preprocess()

# Построение индекса
if not os.path.exists(VECTOR_FILE) or not (os.path.exists(INDEX_FILE)):
    sentences = []
    for doc in documents:
        # Разбиваем на предложения и очищаем
        for sent in re.split(r'(?<=[.!?])\s+', doc):
            sent = clean_text(sent)
            if len(sent.split()) >= 3:  # Игнорируем короткие фрагменты
                sentences.append(sent)
    
    # Удаление дубликатов
    sentences = list(set(sentences))
    
    embeddings = model.encode(sentences, batch_size=32, show_progress_bar=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    with open(VECTOR_FILE, 'wb') as f:
        pickle.dump(sentences, f)
    faiss.write_index(index, INDEX_FILE)
else:
    with open(VECTOR_FILE, 'rb') as f:
        sentences = pickle.load(f)
    index = faiss.read_index(INDEX_FILE)

# Функции поиска
def calculate_relevance(query, fragment, embedding_score):
    """Комбинированная оценка релевантности"""
    text_sim = SequenceMatcher(None, query, fragment).ratio()
    return 0.6 * embedding_score + 0.4 * text_sim  # Взвешенная сумма

def find_context(doc, fragment):
    """Нахождение контекста для фрагмента"""
    words = doc.split()
    frag_words = fragment.split()
    
    for i in range(len(words) - len(frag_words) + 1):
        if words[i:i+len(frag_words)] == frag_words:
            start = max(0, i-5)
            end = min(len(words), i+len(frag_words)+5)
            return ' '.join(words[start:end])
    return fragment

def semantic_search(query, top_k=5, min_score=0.5):
    """Улучшенный семантический поиск"""
    query_processed = clean_text(query)
    query_embed = model.encode([query_processed])
    faiss.normalize_L2(query_embed)
    
    distances, indices = index.search(query_embed, top_k*10)  # Широкий поиск
    
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if score < min_score:
            continue
            
        fragment = sentences[idx]
        relevance = calculate_relevance(query_processed, fragment, score)
        
        # Ищем полный документ, содержащий этот фрагмент
        context = next((doc for doc in documents if fragment in doc), fragment)
        
        results.append({
            'fragment': fragment,
            'context': find_context(context, fragment),
            'similarity': float(relevance),
            'full_text': context
        })
    
    return sorted(results, key=lambda x: -x['similarity'])[:top_k]

# Интерфейс
st.title("Семантический поиск в соцсетях")
query = st.text_input("Введите поисковый запрос:")

col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Количество результатов", 1, 10, 5)
with col2:
    min_score = st.slider("Минимальная релевантность", 0.1, 1.0, 0.5, 0.05)

if st.button("Искать"):
    if query:
        with st.spinner("Поиск..."):
            results = semantic_search(query, top_k, min_score)
            
            if not results:
                st.warning("Нет результатов, удовлетворяющих критериям")
            else:
                st.success(f"Найдено результатов: {len(results)}")
                
                for i, res in enumerate(results, 1):
                    with st.expander(f"Результат #{i} (релевантность: {res['similarity']:.2f})"):
                        st.markdown(f"**Фрагмент:** `{res['fragment']}`")
                        st.markdown("**Контекст:**")
                        st.markdown(res['context'])
                        if st.checkbox("Показать полный текст", key=f"full_{i}"):
                            st.text_area("Полный текст:", res['full_text'], height=150)
    else:
        st.warning("Введите поисковый запрос")