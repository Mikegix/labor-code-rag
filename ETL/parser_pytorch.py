import docx
import re
import json


def parse_labor_code(file_path):
    # 1. Загружаем документ
    doc = docx.Document(file_path)

    parsed_data = []

    # 2. Переменные "состояния" (State)
    current_section = "Не определен"
    current_chapter = "Не определена"

    # Буфер для текущей статьи
    current_article = {
        "number": None,
        "title": None,
        "text_lines": []
    }

    # 3. Регулярные выражения (Regex)
    regex_section = re.compile(r"^Раздел\s+[IVXLC]+\.")  # Пример: Раздел I.
    regex_chapter = re.compile(r"^Глава\s+\d+\.")  # Пример: Глава 1.
    regex_article = re.compile(r"^Статья\s+(\d+(\.\d+)?)\.\s+(.*)$")  # Пример: Статья 15. Текст...

    # 4. Цикл по всем параграфам
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        # Если строка пустая - пропускаем
        if not text:
            continue

        # --- Проверка: Это Раздел? ---
        if regex_section.match(text):
            current_section = text
            continue  # Идем к следующему параграфу

        # --- Проверка: Это Глава? ---
        if regex_chapter.match(text):
            current_chapter = text
            continue

        # --- Проверка: Это начало НОВОЙ Статьи? ---
        match_article = regex_article.match(text)
        if match_article:
            # А. СОХРАНЯЕМ ПРЕДЫДУЩУЮ СТАТЬЮ (если она была)
            if current_article["number"] is not None:
                # Собираем список строк в один текст
                full_text = "\n".join(current_article["text_lines"])

                parsed_data.append({
                    "id": f"art_{current_article['number']}",
                    "article_number": current_article['number'],
                    "title": current_article['title'],
                    "section": current_section,
                    "chapter": current_chapter,
                    "text": full_text,
                    "full_context": f"{current_section} -> {current_chapter} -> Статья {current_article['number']}. {current_article['title']}\n{full_text}"
                })

            # Б. НАЧИНАЕМ НОВУЮ СТАТЬЮ
            article_num = match_article.group(1)  # Номер (15 или 15.1)
            article_name = match_article.group(3)  # Название

            current_article = {
                "number": article_num,
                "title": article_name,
                "text_lines": []  # Текст самой статьи начнем собирать со следующей строки (или этой)
            }


        else:
            # --- Это просто текст внутри статьи ---
            # Добавляем в буфер текущей статьи, только если мы уже внутри какой-то статьи
            if current_article["number"] is not None:
                current_article["text_lines"].append(text)

    # 5. ВАЖНО: Сохраняем самую последнюю статью после выхода из цикла
    if current_article["number"] is not None:
        full_text = "\n".join(current_article["text_lines"])
        parsed_data.append({
            "id": f"art_{current_article['number']}",
            "article_number": current_article['number'],
            "title": current_article['title'],
            "section": current_section,
            "chapter": current_chapter,
            "text": full_text,
            "full_context": f"{current_section} -> {current_chapter} -> Статья {current_article['number']}. {current_article['title']}\n{full_text}"
        })

    return parsed_data


# Запуск
data = parse_labor_code("TK_RF.docx")

# Сохранение в JSON
with open("labor_code_processed.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Готово! Обработано {len(data)} статей.")