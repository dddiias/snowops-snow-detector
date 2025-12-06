# Snow Detector

Сервис отслеживает грузовики в RTSP-потоке, оценивает объём снега на кузове через Gemini и отправляет событие в SnowOps backend вместе со снимком.

## Быстрый запуск
- Python 3.10+, ffmpeg в `PATH` (для RTSP через OpenCV), файл весов `yolov8n.pt` рядом с `main.py`.
- Установить зависимости:
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
- Скопировать `sample_env` в `.env` и прописать `GEMINI_API_KEY=<ваш ключ>`.
- Запуск: `python main.py` (закрыть окно — `q` или `Esc`).

## Настройка
Все настраиваемые параметры в `main.py`:
- `VIDEO_SOURCE_URL` — RTSP-адрес камеры.
- `YOLO_MODEL_PATH` — путь к весам YOLOv8.
- `TRUCK_CLASS_ID` / `CONFIDENCE_THRESHOLD` — класс «truck» (COCO: 7) и порог уверенности.
- `CENTER_ZONE_START_X` / `CENTER_ZONE_END_X` / `CENTER_LINE_X` — доли ширины кадра, задающие центральную зону и контрольную линию.
- `MIN_DIRECTION_DELTA` — минимальное смещение центра bbox по X между кадрами, чтобы считать движение слева направо.
- `SNAPSHOT_BASE_DIR` — директория для снимков и JSON с результатами.
- `GEMINI_MODEL` — модель Gemini для анализа.
- `BACKEND_ENDPOINT` / `CAMERA_ID` — куда и с каким ID отправлять событие в SnowOps.

## Логика работы
1. Открывается RTSP-поток `VIDEO_SOURCE_URL` и создаётся окно отображения (`DISPLAY_WIDTH/HEIGHT`).
2. На каждом кадре YOLOv8 ищет грузовики (class 7). Берётся bbox с максимальной площадью и уверенностью ≥ `CONFIDENCE_THRESHOLD`.
3. Центр bbox сверяется с центральной зоной (`CENTER_ZONE_START_X..END_X`). Движение слева направо фиксируется, если сдвиг центра > `MIN_DIRECTION_DELTA`.
4. При первом срабатывании условия «грузовик в зоне и движется слева направо»:
   - Сохраняется кадр `snapshots/YYYY-MM-DD/HH-MM-SS.jpg`.
   - Gemini (`GEMINI_MODEL`) получает снимок и возвращает JSON вида `{percentage, confidence, direction}`; если формат иной, оригинал пишется в `raw`.
   - В `snapshots/.../HH-MM-SS.json` сохраняются `timestamp`, `image_path`, `gemini`.
   - Отправляется событие в SnowOps backend.
5. Пока грузовик виден, повторные события не шлются; флаг сбрасывается, когда детекции нет.

## Что отправляется в SnowOps
POST `multipart/form-data` на `BACKEND_ENDPOINT`.
- Поле `event`: JSON-строка
```json
{
  "camera_id": "camera-001",
  "event_time": "2025-12-06T12:34:56Z",
  "snow_volume_percentage": 42,
  "snow_volume_confidence": 0.91,
  "snow_direction_ai": "left_to_right"
}
```
  - `event_time` — время фиксации кадра в локальном времени машины (микросекунды отброшены); суффикс `Z` просто добавляется, перевод в UTC не выполняется.
  - `snow_volume_*` могут быть `null`, если ответ Gemini не разобрался.
  - `snow_direction_ai` принимает `left_to_right` / `right_to_left` / `unknown`.
- Поле(я) `photos`: один или несколько файлов `image/jpeg` с кадром(ами) из `snapshots/` (сервис отправляет один).

Пример ручной отправки сохранённого кадра:
```bash
curl -X POST https://snowops-anpr-service.onrender.com/api/v1/anpr/events ^
  -F "event={\"camera_id\":\"camera-001\",\"event_time\":\"2025-12-06T12:34:56Z\",\"snow_volume_percentage\":42,\"snow_volume_confidence\":0.91,\"snow_direction_ai\":\"left_to_right\"}" ^
  -F "photos=@snapshots/2025-12-06/12-34-56.jpg;type=image/jpeg"
```

## Полезное
- Все снимки и результаты анализа складываются в `snapshots/` (каталог исключён из git).
- При проблемах с RTSP поток переподключается после 50 неудачных чтений кадров.
- Завершение работы — закрытие окна, `q` или `Esc`.
