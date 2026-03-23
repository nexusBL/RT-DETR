import cv2
import torch
import time
import random
from ultralytics import YOLO

MODEL_PATH = r'C:\Users\icroc\Documents\RT-DETR\BEST_MODEL_QCAR2.pt'  # change this to your path

CONF   = 0.40
CAMERA = 0

print('Loading model...')
model  = YOLO(MODEL_PATH)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device}')

class_names = model.names
random.seed(42)
COLORS = {i: tuple(random.randint(80, 255) for _ in range(len(class_names)))[0:3] for i in range(len(class_names))}
COLORS = {i: tuple(random.randint(80, 255) for _ in range(3)) for i in range(len(class_names))}

cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print('Camera not found! Try changing CAMERA = 1')
    exit()

print('Running! Press Q to quit, S to save screenshot')
prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source  = frame,
        conf    = CONF,
        imgsz   = 640,
        device  = device,
        verbose = False,
        half    = (device == 'cuda'),
    )

    result = results[0]
    if result.boxes is not None and len(result.boxes) > 0:
        boxes   = result.boxes.xyxy.cpu().numpy()
        confs   = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            color = COLORS[cls_id]
            label = class_names[cls_id]
            text  = f'{label}  {conf:.0%}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    curr = time.time()
    fps  = 1.0 / (curr - prev + 1e-9)
    prev = curr
    cv2.putText(frame, f'FPS: {fps:.1f}  |  Conf: {CONF}  |  Q=Quit  S=Screenshot',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Traffic Sign Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        from datetime import datetime
        fname = f'screenshot_{datetime.now().strftime("%H%M%S")}.jpg'
        cv2.imwrite(fname, frame)
        print(f'Saved: {fname}')

cap.release()
cv2.destroyAllWindows()