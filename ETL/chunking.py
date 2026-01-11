import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Настройка параметров нарезки
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]  # Приоритет разделителей
)


def process_chunks(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    chunked_docs = []

    for article in articles:
        # Берем "чистый" текст для эмбеддинга
        text_content = article.get("text_for_embedding", article["text"])

        # Метаданные, которые пойдут в каждый кусок
        base_metadata = {
            "article_id": article["id"],
            "article_number": article["article_number"],
            "title": article["title"],
            "chapter": article["chapter"],
            "section": article["section"],
            "original_full_text": article.get("text_for_llm", article["text"])
        }

        # --- Логика разделения ---
        chunks = splitter.create_documents([text_content], metadatas=[base_metadata])

        for i, chunk in enumerate(chunks):
            doc_record = chunk.metadata.copy()  # Копируем метаданные
            doc_record["chunk_text"] = chunk.page_content  # Текст кусочка
            doc_record["chunk_index"] = i  # Порядковый номер куска внутри статьи (0, 1, 2...)

            # Создаем уникальный ID для Векторной Базы
            doc_record["vector_id"] = f"{article['id']}_{i}"

            chunked_docs.append(doc_record)

    print(f"Было статей: {len(articles)}")
    print(f"Стало чанков: {len(chunked_docs)}")

    # Сохраняем результат
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunked_docs, f, ensure_ascii=False, indent=4)


# Запуск
process_chunks("labor_code_processed.json", "labor_code_chunks.json")