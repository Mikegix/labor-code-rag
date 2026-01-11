import streamlit as st
import requests
import os

# Настройка страницы
st.set_page_config(page_title="Юрист-Бот ТК РФ", page_icon="⚖️")
st.title("⚖️ ИИ-Консультант по Трудовому Кодексу")

# URL нашего API (который мы запустили на шаге выше)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")

# 1. Инициализация истории чата
# Streamlit перезагружает скрипт при каждом клике. 
# session_state позволяет запомнить переписку между перезагрузками.
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Отрисовка истории сообщений
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Поле ввода вопроса (внизу страницы)
if prompt := st.chat_input("Задайте вопрос по трудовому праву..."):
    # А. Отображаем сообщение пользователя
    with st.chat_message("user"):
        st.markdown(prompt)
    # Добавляем в историю
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Б. Получаем ответ от API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # Место, где появится текст
        message_placeholder.markdown("⏳ *Изучаю документы...*")

        try:
            # Отправляем запрос на наш FastAPI сервер
            response = requests.post(API_URL, json={"query": prompt})

            if response.status_code == 200:
                data = response.json()
                answer_text = data["answer"]
                sources = data["sources"]

                # Формируем красивый вывод с источниками
                full_response = f"{answer_text}\n\n---\n**Источники:**"
                for src in sources:
                    full_response += f"\n- {src}"

                message_placeholder.markdown(full_response)

                # Сохраняем ответ бота в историю
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                message_placeholder.error(f"Ошибка сервера: {response.status_code}")

        except Exception as e:
            message_placeholder.error(f"Не удалось подключиться к API. Сервер запущен? Ошибка: {e}")