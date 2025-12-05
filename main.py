import cv2
import os
import time
import json
import numpy as np
from ultralytics import YOLO

VIDEO_SOURCE_URL = "rtsp://admin:Armat456321@194.26.239.249:555/Streaming/Channels/101" 
YOLO_MODEL_PATH = "yolov8n.pt" 

TRUCK_CLASS_ID = 7 
CONFIDENCE_THRESHOLD = 0.55 
CENTER_ZONE_START_X = 0.4 
CENTER_ZONE_END_X = 0.6 
SNAPSHOT_DIR = "snapshots"
DISPLAY_WIDTH = 1280 
DISPLAY_HEIGHT = 720

HISTORY_LENGTH = 15 
MAX_HISTORY = 50 

if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"✅ Модель YOLO загружена: {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"❌ Ошибка загрузки модели YOLO: {e}")
    exit()

def get_cargo_bbox(truck_bbox: list, direction: str) -> list:
    x1, y1, x2, y2 = map(int, truck_bbox)
    w = x2 - x1
    h = y2 - y1
    
    CAB_CUT_RATIO = 0.38
    VERTICAL_PAD = 0.08

    y1_cargo = y1 + int(h * VERTICAL_PAD)
    y2_cargo = y2 - int(h * VERTICAL_PAD)

    if direction == "Вправо":
        x1_cargo = x1
        x2_cargo = x2 - int(w * CAB_CUT_RATIO) 
    else: 
        x1_cargo = x1 + int(w * CAB_CUT_RATIO)
        x2_cargo = x2
        
    return [x1_cargo, y1_cargo, x2_cargo, y2_cargo]


def calculate_snow_volume(frame: np.ndarray, cargo_bbox: list) -> float:
    cx1, cy1, cx2, cy2 = map(int, cargo_bbox)
    
    cargo_area = frame[cy1:cy2, cx1:cx2]
    
    if cargo_area.size == 0:
        return 0.0

    hsv_area = cv2.cvtColor(cargo_area, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([179, 70, 255])
    snow_mask = cv2.inRange(hsv_area, lower_white, upper_white)
    
    snow_points = cv2.findNonZero(snow_mask)
    
    if snow_points is None:
        return 0.0 

    min_y_snow_rel = np.min(snow_points[:, :, 1]) 
    
    H_cargo = cy2 - cy1
    H_snow = H_cargo - min_y_snow_rel
    
    volume_ratio = H_snow / H_cargo
    
    return min(max(volume_ratio, 0.0), 1.0) 


def determine_direction_by_history(history: list) -> str:
    if len(history) < HISTORY_LENGTH:
        return "Неизвестно"
    
    first_x = history[0]
    current_x = history[-1]
    
    MOVEMENT_THRESHOLD = 5 
    
    if current_x - first_x > MOVEMENT_THRESHOLD:
        return "Вправо"
    elif first_x - current_x > MOVEMENT_THRESHOLD:
        return "Влево"
    else:
        return "Стоит" 


def process_video_stream():
    cap = cv2.VideoCapture(VIDEO_SOURCE_URL)

    if not cap.isOpened():
        print(f"Не работает хуйня {VIDEO_SOURCE_URL}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    center_start_pixel = int(frame_width * CENTER_ZONE_START_X)
    center_end_pixel = int(frame_width * CENTER_ZONE_END_X)
    
    print(f"центр {center_start_pixel} {center_end_pixel}px")

    current_truck_history = [] 
    truck_has_been_processed = False 

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        results = yolo_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, classes=[TRUCK_CLASS_ID])
        truck_bbox = None
        
        if results and results[0].boxes and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            truck_bbox = box.xyxy[0].tolist()
        
        center_x = frame_width // 2 
        cv2.line(frame, (center_x, 0), (center_x, frame_height), (0, 255, 255), 1) 
        cv2.line(frame, (center_start_pixel, 0), (center_start_pixel, frame_height), (0, 255, 0), 2)
        cv2.line(frame, (center_end_pixel, 0), (center_end_pixel, frame_height), (0, 255, 0), 2)

        if truck_bbox:
            x1, y1, x2, y2 = truck_bbox
            w = x2 - x1
            object_center_x = x1 + w // 2
            
            current_truck_history.append(object_center_x)
            if len(current_truck_history) > MAX_HISTORY:
                 current_truck_history.pop(0) 
            
            direction = determine_direction_by_history(current_truck_history)

            is_in_center = center_start_pixel < object_center_x < center_end_pixel
            is_moving_right = direction == "Вправо"
            is_direction_known = direction != "Неизвестно"

            if is_in_center and is_moving_right and is_direction_known and not truck_has_been_processed:
                
                cargo_bbox = get_cargo_bbox(truck_bbox, direction)
                cx1, cy1, cx2, cy2 = cargo_bbox
                volume = calculate_snow_volume(frame, cargo_bbox)
                
                timestamp = time.strftime("%Y%m%d%H%M%S")
                snapshot_path = os.path.join(SNAPSHOT_DIR, f"capture_{timestamp}.jpg")
                
                result_text = f"Volume: {round(volume * 100)}% | Dir: {direction}"
                cv2.putText(frame, result_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (0, 255, 255), 2) 

                cv2.imwrite(snapshot_path, frame)
                
                analysis_result = {
                    "timestamp": timestamp,
                    "НаправлениеКабины": direction,
                    "ОценкаОбъема": round(volume, 2),
                    "Комментарий": f"Объем: {round(volume*100)}%, Направление: {direction}",
                }
                
                print(f"\n--- Грузовик (Направление: {direction}) захвачен в центре: {timestamp} ---")
                print(json.dumps(analysis_result, indent=4, ensure_ascii=False))
                
                truck_has_been_processed = True 
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"Tracking: {direction}", (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        else:
            current_truck_history = []
            truck_has_been_processed = False 
        
        resized_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow('Video Stream Analysis', resized_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(f"пашет,пашет")
    process_video_stream()