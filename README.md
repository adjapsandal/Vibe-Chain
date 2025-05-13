# YouTube Content-Based Recommendation System

## Содержание

1. [Стек технологий и зависимости](#стек-технологий-и-зависимости)  
2. [Структура проекта](#структура-проекта)  
3. [Установка и настройка](#установка-и-настройка)  
4. [Описание модулей](#описание-модулей)  
   - `fetch_metadata.py`  
   - `transcript.py`  
   - `embedder.py`  
   - `recommend.py`  
5. [Запуск и примеры](#запуск-и-примеры)  
6. [Формат корпуса данных](#формат-корпуса-данных)  
7. [Планы на будущее](#планы-на-будущее)  

## Стек технологий и зависимости

- Python 3.8+  
- YouTube Data API v3 (`google-api-python-client`)  
- `youtube-transcript-api` для извлечения субтитров  
- `sentence-transformers` (SBERT)  
- `transformers` (CLIP)  
- `pandas`, `numpy`, `scikit-learn` (косинусное сходство)  
- `requests`, `pytube`, `Pillow` для работы с изображениями и видео  
- `tqdm` для индикаторов прогресса  
- `python-dotenv` для управления переменными окружения  

Список всех зависимостей в файле [`requirements.txt`](requirements.txt).
## Установка и настройка

1. **Клонировать репозиторий**  
   ```bash
   git clone https://github.com/adjapsandal/Vibe-Chain.git
   cd vibe_chain
   ```

2. **Виртуальное окружение**
    
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # Linux/Mac
    
    venv\Scripts\activate      # Windows
    ```
    
3. **Установить зависимости**
    
    ```bash
    pip install -r requirements.txt
    ```
    
4. **YouTube API Key**
    
    - Создать проект в Google Cloud Console
        
    - Включить YouTube Data API v3
        
    - Создать API-ключ и вписать его в файл `.env` рядом с корнем репозитория:
        
        ```
        YOUTUBE_API_KEY=ваш_ключ
        ```

## Описание модулей

### `fetch_metadata.py`

- **`extract_video_id(url)`**  
    Разбирает любые форматы YouTube-ссылок и возвращает чистый `video_id`.
    
- **`get_video_snippet(id_video)`**  
    Через YouTube Data API получает:
    
    - `title`
        
    - `description`
        
    - `tags`
        
    - `thumbnail_url
        
    
    Возвращает `None`, если видео не найдено.

### `transcript.py`

- Использует `youtube-transcript-api` для получения субтитров:
    
    1. Сначала пытается найти ручные субтитры (`find_manually_created_transcript`),
        
    2. затем автоматические (`find_generated_transcript`),
        
    3. в случае необходимости переводит в английский (`.translate('en')`).
        
- **`get_transcript(video_id)`**  
    Возвращает весь текст субтитров одной строкой.  
    При ошибках (нет субтитров, видео удалено) возвращает пустую строку и логирует причину.

### `embedder.py`

- **`encode_text(text)`**  
    SBERT (`all-mpnet-base-v2`) для текста, нормализует вектор.
    
- **`encode_image(image)`**  
    CLIP-ViT (`openai/clip-vit-base-patch32`), нормализует результат.
    
- **`embed_video(url)`**
    
    1. Извлекает `video_id` через `extract_video_id`
        
    2. Собирает `meta_str = title + description + tags…`
        
    3. Получает `transcript`
        
    4. Считает `text_emb` + `img_emb` (миниатюра по URL)
        
    5. Конкатенирует векторы `[768 + 512]`

### `recommend.py`

- **Параметры CLI**
    
    - `video_id` (позиционный) — ID входного видео
        
    - `--corpus` — путь к CSV (по умолчанию `data/small_trending.csv`)
        
    - `--cache` — путь к pickle для кеша эмбеддингов
        
    - `--rebuild` — пересчитать эмбеддинги заново
        
    - `--topk` — число рекомендаций (по умолчанию 5)
        
- **Логика**
    
    1. Если указали `--rebuild` или кеш отсутствует, вызывается `build_embeddings()`
        
    2. `load_embeddings()` подгружает `(video_ids, embs)`
        
    3. Для входного `video_id` строится эмбеддинг «на лету» через `embed_video()`
        
    4. Косинусное сходство с матрицей `embs`, сортировка и вывод топ-K ссылок
        
- **Пример запуска**
    
    ```bash
    # пересчёт эмбеддингов + рекомендации для dQw4w9WgXcQ
    python recommend.py dQw4w9WgXcQ --rebuild
    
    # 10 рекомендаций без пересчёта
    python recommend.py dQw4w9WgXcQ --topk 10
    ```

## Формат корпуса данных

Ваш CSV (`data/small_trending.csv`) должен содержать одну колонку:

```csv
video_id
dQw4w9WgXcQ
abcd1234EfG
...
```

Если вам нужно явно хранить URL, добавьте вторую колонку `url`:

```csv
video_id,url
dQw4w9WgXcQ,https://youtu.be/dQw4w9WgXcQ
abcd1234EfG,https://www.youtube.com/watch?v=abcd1234EfG
```

При наличии `url` скрипт в `build_embeddings` будет брать именно его, иначе строит `https://youtu.be/{video_id}` автоматически.

---

## Планы на будущее

- **Faiss** для быстрого поиска соседей в корпусе из сотен тысяч видео
    
- **Ключевые кадры** вместо одной миниатюры, усреднённый визуальный эмбеддинг
    
- **Тонкая настройка моделей** (изменить SBERT/CLIP)
    
- **Веб-интерфейс** (FastAPI + Django)
    
- **Работа с пользовательскими сигналами** (плейлисты, лайки, история просмотров)
    
- **Сохранение и логирование** результатов в БД (PostgreSQL)
