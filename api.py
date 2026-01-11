from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_generation import ask_labor_code  # Импортируем вашу функцию

# 1. Создаем приложение
app = FastAPI(
    title="Labor Code RAG API",
    description="API для консультаций по Трудовому Кодексу РФ на базе Gemma 3",
    version="1.0.0"
)


# 2. Описываем структуру запроса
class QueryRequest(BaseModel):
    query: str


# 3. Описываем структуру ответа
class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


# 4. Создаем эндпоинт
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        # Вызываем логику RAG
        answer, sources = ask_labor_code(request.query)

        return QueryResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        # Если что-то сломалось, возвращаем 500 ошибку
        raise HTTPException(status_code=500, detail=str(e))


# 5. Простой эндпоинт для проверки, жив ли сервер
@app.get("/health")
async def health_check():
    return {"status": "ok"}