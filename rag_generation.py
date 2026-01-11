import chromadb
from chromadb.utils import embedding_functions
import ollama
import os

ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

print(f"DEBUG: Подключаюсь к Ollama по адресу: {ollama_host}")

client = ollama.Client(host=ollama_host)

# --- КОНФИГУРАЦИЯ ---
DB_PATH = "./chroma_db_data"
COLLECTION_NAME = "labor_code"
MODEL_NAME = "gemma3:4b"

# 1. Подключение к Базе Данных
client = chromadb.PersistentClient(path=DB_PATH)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)


def get_context(query, n_results=3):
    """
    Ищет в базе и возвращает склеенный текст статей для промпта.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    context_parts = []
    sources = []

    # Разбираем результаты
    if results['documents']:
        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i]

            # ВАЖНО: Мы берем 'original_full_text', который сохранили в метаданных!
            # Это даст модели полный текст статьи, а не обрывок.
            full_text = meta.get('original_full_text', results['documents'][0][i])

            # Формируем красивый блок для контекста
            source_info = f"Статья {meta['article_number']}: {meta['title']}"
            context_part = f"ИСТОЧНИК: {source_info}\nТЕКСТ:\n{full_text}"

            context_parts.append(context_part)
            sources.append(source_info)

    return "\n\n---\n\n".join(context_parts), sources


def ask_labor_code(question):
    print(f"\n🤖 Думаю над вопросом: '{question}'...")

    # 1. Retrieval (Поиск)
    context_text, sources = get_context(question)

    if not context_text:
        return "К сожалению, я не нашел информации в базе знаний.", []

    # 2. Augmented Generation (Промпт)
    prompt = f"""
Ты — профессиональный юрист-консультант по Трудовому Кодексу РФ.
Твоя задача — ответить на вопрос пользователя, используя ТОЛЬКО предоставленный ниже контекст.

Инструкции:
1. Используй только факты из раздела "КОНТЕКСТ". Не придумывай законы, которых нет в тексте.
2. Ссылайся на номера статей. Например: "Согласно ст. 261 ТК РФ...".
3. Если в контексте нет ответа на вопрос, честно напиши: "В предоставленных документах нет информации об этом".
4. Ответ должен быть кратким, четким и вежливым.

КОНТЕКСТ:
{context_text}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}
"""

    # 3. Generation (Вызов Ollama)
    response = ollama.chat(model=MODEL_NAME, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    return response['message']['content'], sources


# --- ЗАПУСК ---
if __name__ == "__main__":
    while True:
        user_query = input("\nВведите ваш вопрос (или 'exit'): ")
        if user_query.lower() in ['exit', 'quit', 'выход']:
            break

        answer, found_sources = ask_labor_code(user_query)

        print("\n" + "=" * 50)
        print("ОТВЕТ ЮРИСТА:")
        print(answer)
        print("=" * 50)
        print("Использованные источники:")
        for s in found_sources:
            print(f"- {s}")