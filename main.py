import os
import json
from datetime import datetime
import time

import cv2
import numpy as np
import requests
from ultralytics import YOLO
from google import genai
from PIL import Image
from dotenv import load_dotenv

# ============== ОКРУЖЕНИЕ ==============

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError(
        "Переменная GEMINI_API_KEY не найдена. "
        "Создай файл .env с строкой: GEMINI_API_KEY=ТВОЙ_КЛЮЧ"
    )

gemini_client = genai.Client()

# ============== НАСТРОЙКИ ==============

VIDEO_SOURCE_URL = "rtsp://admin:Armat456321@194.26.239.249:555/Streaming/Channels/101"
YOLO_MODEL_PATH = "yolov8n.pt"

TRUCK_CLASS_ID = 7
CONFIDENCE_THRESHOLD = 0.55

CENTER_ZONE_START_X = 0.35
CENTER_ZONE_END_X = 0.65
CENTER_LINE_X = 0.5  # жёлтая линия

SNAPSHOT_BASE_DIR = "snapshots"

# размер окна отображения
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# порог для «движется вправо»
MIN_DIRECTION_DELTA = 5

GEMINI_MODEL = "gemini-2.5-flash"

# бекенд SnowOps
BACKEND_ENDPOINT = "https://snowops-anpr-service.onrender.com/api/v1/anpr/events"
CAMERA_ID = "camera-001"   # поменяешь на реальный ID камеры

# =======================================


def init_model() -> YOLO:
    return YOLO(YOLO_MODEL_PATH)


def detect_truck_bbox(frame: np.ndarray, model: YOLO):
    """
    Находит bbox грузовика (truck).
    Возвращает (x1, y1, x2, y2) или None.
    """
    results = model(frame, verbose=False)
    best_box = None
    best_area = 0.0

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for b in boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())

            if cls_id != TRUCK_CLASS_ID or conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)

            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

    return best_box


def check_center_zone(bbox, frame_width: int):
    """
    Проверяет, попал ли центр bbox в центральный коридор.
    """
    x1, y1, x2, y2 = bbox
    center_x = x1 + (x2 - x1) // 2

    zone_start_px = int(frame_width * CENTER_ZONE_START_X)
    zone_end_px = int(frame_width * CENTER_ZONE_END_X)

    in_zone = zone_start_px < center_x < zone_end_px
    return in_zone, center_x, zone_start_px, zone_end_px


_last_center_x = None


def is_moving_left_to_right(current_center_x: int) -> bool:
    """
    True, если объект движется слева направо.
    ЛОГИКА КАК В РАБОТАВШЕМ ВАРИАНТЕ.
    """
    global _last_center_x

    moved_right = False
    if _last_center_x is not None:
        if current_center_x - _last_center_x > MIN_DIRECTION_DELTA:
            moved_right = True

    _last_center_x = current_center_x
    return moved_right


def save_frame(frame: np.ndarray):
    """
    Сохранить кадр в snapshots/YYYY-MM-DD/HH-MM-SS.jpg.
    """
    now = datetime.now()
    date_dir = os.path.join(SNAPSHOT_BASE_DIR, now.strftime("%Y-%m-%d"))
    os.makedirs(date_dir, exist_ok=True)

    filename = now.strftime("%H-%M-%S") + ".jpg"
    path = os.path.join(date_dir, filename)

    cv2.imwrite(path, frame)
    return path, now


def analyze_snow_gemini(image_path: str) -> dict:
    """
    Анализ объёма снега и направления движения по картинке через Gemini.
    Возвращает dict, при ошибке — {"error": "..."}.
    """
    try:
        image = Image.open(image_path)

        prompt = (
            "На изображении находится грузовой автомобиль (КАМАЗ или похожий) с кузовом.\n"
            "1) Оцени, на сколько процентов от объёма кузов заполнен снегом (0-100).\n"
            "2) Определи НАПРАВЛЕНИЕ движения грузовика по дорожным следам, положению колёс и фону.\n"
            "Возможные значения направления:\n"
            '  - \"left_to_right\" — если грузовик едет слева направо\n'
            '  - \"right_to_left\" — если грузовик едет справа налево\n'
            '  - \"unknown\" — если направление определить нельзя\n\n'
            "Важно: верни СТРОГО один JSON-объект БЕЗ ``` и любого лишнего текста:\n"
            '{\n'
            '  "percentage": 0,\n'
            '  "confidence": 0.0,\n'
            '  "direction": "left_to_right"\n'
            "}\n"
        )

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[image, prompt],
        )

        text = (response.text or "").strip()

        # на всякий случай срезаем ```json ... ```
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            data = json.loads(text)
        except Exception:
            data = {"raw": text}

        return data

    except Exception as e:
        print(f"[GEMINI] error: {e}")
        return {"error": str(e)}


def save_analysis_json(image_path: str, timestamp: datetime, gemini_result: dict) -> str:
    """
    Сохранить результат анализа рядом с изображением.
    """
    json_path = image_path.rsplit(".", 1)[0] + ".json"

    payload = {
        "timestamp": timestamp.isoformat(),
        "image_path": image_path,
        "gemini": gemini_result,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return json_path


def _extract_gemini_fields(gemini_result: dict):
    """
    Достаём percentage, confidence, direction из ответа Gemini.
    Умеет парсить и поле raw с ```json ...``` при необходимости.
    """
    percentage = None
    confidence = None
    direction = None

    if not isinstance(gemini_result, dict):
        return percentage, confidence, direction

    p = gemini_result.get("percentage")
    c = gemini_result.get("confidence")
    d = gemini_result.get("direction")
    raw = gemini_result.get("raw")

    # если чего-то нет — пробуем распарсить raw
    if (p is None or c is None or d is None) and raw:
        raw_s = str(raw).strip()
        try:
            if raw_s.startswith("```"):
                raw_s = raw_s.strip("`")
                if raw_s.lower().startswith("json"):
                    raw_s = raw_s[4:].strip()
            parsed = json.loads(raw_s)
            if p is None:
                p = parsed.get("percentage")
            if c is None:
                c = parsed.get("confidence")
            if d is None:
                d = parsed.get("direction")
        except Exception:
            pass

    try:
        if p is not None:
            percentage = int(round(float(p)))
    except Exception:
        pass

    try:
        if c is not None:
            confidence = float(c)
    except Exception:
        pass

    if d is not None:
        direction = str(d).strip().lower()

    return percentage, confidence, direction


def send_event_to_backend(image_paths, gemini_result: dict, timestamp: datetime):
    """
    Отправляет событие на SnowOps backend.
    Если Gemini говорит, что машинка НЕ слева-направо —
    событие не отправляем.
    """
    percentage, confidence, direction = _extract_gemini_fields(gemini_result)

    event_payload = {
        "camera_id": CAMERA_ID,
        "event_time": timestamp.replace(microsecond=0).isoformat() + "Z",
        "snow_volume_percentage": percentage,
        "snow_volume_confidence": confidence,
        # просто логируем, но не используем как фильтр
        "snow_direction_ai": direction,
    }

    # Формируем files для multipart/form-data
    files = []
    file_handles = []
    for path in image_paths:
        try:
            f = open(path, "rb")
            file_handles.append(f)
            files.append(("photos", (os.path.basename(path), f, "image/jpeg")))
        except Exception as e:
            print(f"[UPSTREAM] warning: cannot open file {path}: {e}")

    data = {"event": json.dumps(event_payload, ensure_ascii=False)}

    try:
        resp = requests.post(
            BACKEND_ENDPOINT,
            data=data,
            files=files,
            timeout=15,
        )
        status = resp.status_code
        body = resp.text.strip().replace("\n", "")
        print(f"[UPSTREAM] status={status}, body={body}")
        return status, body
    except Exception as e:
        print(f"[UPSTREAM] network_error={e}")
        return None, str(e)
    finally:
        for f in file_handles:
            try:
                f.close()
            except Exception:
                pass


event_sent_for_current_truck = False


def process_video_stream():
    global event_sent_for_current_truck, _last_center_x

    model = init_model()

    cap = cv2.VideoCapture(VIDEO_SOURCE_URL)
    if not cap.isOpened():
        print("❌ Не удалось открыть видеопоток:", VIDEO_SOURCE_URL)
        return

    window_name = "Video Stream Analysis"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    print("✅ Старт анализа видеопотока... Нажми 'q' для выхода.")

    frame_width = None
    frame_height = None
    center_start_pixel = None
    center_end_pixel = None
    center_x_geom = None

    fail_count = 0
    MAX_FAILS = 50  # после 50 подряд неудач — переподключаемся

    while True:
        ret, frame = cap.read()

        # сначала проверяем, что кадр живой
        if not ret or frame is None or frame.size == 0:
            fail_count += 1
            print(f"⚠️ Не удалось прочитать кадр (fail={fail_count})")

            if fail_count >= MAX_FAILS:
                print("🔁 Слишком много ошибок чтения, переподключаемся к камере...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(VIDEO_SOURCE_URL)
                fail_count = 0

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            continue

        # теперь можно делать копию «сытого» кадра
        raw_frame = frame.copy()
        fail_count = 0

        if frame_width is None:
            frame_height, frame_width = frame.shape[:2]
            center_start_pixel = int(frame_width * CENTER_ZONE_START_X)
            center_end_pixel = int(frame_width * CENTER_ZONE_END_X)
            center_x_geom = int(frame_width * CENTER_LINE_X)
            print(f"Центральный коридор: {center_start_pixel}px .. {center_end_pixel}px")

        # линии
        cv2.line(frame, (center_x_geom, 0), (center_x_geom, frame_height),
                 (0, 255, 255), 1)
        cv2.line(frame, (center_start_pixel, 0), (center_start_pixel, frame_height),
                 (0, 255, 0), 2)
        cv2.line(frame, (center_end_pixel, 0), (center_end_pixel, frame_height),
                 (0, 255, 0), 2)

        # детекция делаем по raw_frame (без линий и прямоугольников)
        truck_bbox = detect_truck_bbox(raw_frame, model)

        if truck_bbox:
            in_zone, center_x_obj, _, _ = check_center_zone(truck_bbox, frame_width)
            x1, y1, x2, y2 = truck_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            moving_right = is_moving_left_to_right(center_x_obj)
            print(f"[DBG] center_x={center_x_obj}, last_center={_last_center_x}, "
                f"moving_right={moving_right}, in_zone={in_zone}")


            # сработать только ОДИН раз за проход
            if in_zone and moving_right and not event_sent_for_current_truck:
                print("🚛 КамАЗ в коридоре и движется слева-направо — сохраняем и анализируем.")
                image_path, ts = save_frame(raw_frame)
                print("💾 Кадр сохранён:", image_path)

                gemini_result = analyze_snow_gemini(image_path)
                print("📊 Результат Gemini:", gemini_result)

                save_analysis_json(image_path, ts, gemini_result)

                send_event_to_backend([image_path], gemini_result, ts)

                event_sent_for_current_truck = True

        else:
            # грузовик пропал — готовимся к новому событию
            event_sent_for_current_truck = False
            _last_center_x = None

        resized_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow(window_name, resized_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Работа завершена.")


if __name__ == "__main__":
    process_video_stream()
