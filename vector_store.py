import json
import chromadb
from chromadb.utils import embedding_functions

# 1. Настройка путей
INPUT_FILE = "ETL/labor_code_chunks.json"
DB_PATH = "./chroma_db_data"
COLLECTION_NAME = "labor_code"


def create_vector_db():
    # 2. Инициализация Клиента ChromaDB
    # PersistentClient сохраняет данные на диск.
    client = chromadb.PersistentClient(path=DB_PATH)

    # 3. Настройка Эмбеддинг-функции
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 4. Создание (или получение) коллекции
    # get_or_create позволяет запускать скрипт много раз без ошибок
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}  # Используем косинусное сходство для поиска
    )

    # 5. Подготовка данных
    print("Загрузка JSON...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    ids = []
    documents = []
    metadatas = []

    print(f"Подготовка {len(chunks)} документов...")

    for item in chunks:
        # ID должен быть уникальной строкой
        ids.append(item["vector_id"])

        # То, что мы превращаем в вектор (чистый текст статьи)
        documents.append(item["chunk_text"])

        # Метаданные.
        # Поэтому оставляем только плоские поля (строки, числа).
        meta = {
            "article_number": item["article_number"],
            "title": item["title"],
            "chapter": item["chapter"],
            "section": item["section"],
            # Важно: сохраняем полный текст оригинала в метаданных,
            # чтобы потом достать его для LLM.
            "original_full_text": item["original_full_text"]
        }
        metadatas.append(meta)

    # 6. Загрузка в базу (пачками)
    batch_size = 100
    total_batches = len(ids) // batch_size + (1 if len(ids) % batch_size > 0 else 0)

    print(f"Начинаем загрузку в БД ({total_batches} батчей)...")

    for i in range(0, len(ids), batch_size):
        # Срез текущего батча
        batch_ids = ids[i: i + batch_size]
        batch_docs = documents[i: i + batch_size]
        batch_meta = metadatas[i: i + batch_size]

        # Добавляем в коллекцию.
        # Chroma сама вызовет emb_fn, превратит текст в векторы и сохранит.
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta
        )
        print(f"Загружен батч {i // batch_size + 1}/{total_batches}")

    print(f"Успешно! База сохранена в папку '{DB_PATH}'")
    print(f"Всего документов в коллекции: {collection.count()}")


if __name__ == "__main__":
    create_vector_db()