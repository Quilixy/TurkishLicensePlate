import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
from collections import deque, Counter

# YOLO modelini yükle (kendi eğittiğin model dosyası)
model = YOLO("best.pt")

# EasyOCR
reader = easyocr.Reader(['en'])

# Kamera başlat
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

# Kararlı plaka tanıma için buffer
recent_plates = deque(maxlen=10)
stable_plate = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # YOLO ile plaka tespiti
    results = model.predict(source=frame, conf=0.5, classes=0, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        plate_img = frame[y1:y2, x1:x2]

        if plate_img.size == 0:
            continue

        # Görüntü iyileştirme
        plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # OCR işlemi
        ocr_result = reader.readtext(thresh)
        sorted_text = sorted(ocr_result, key=lambda x: x[0][0][0])
        plate_text = "".join([text for (_, text, prob) in sorted_text if prob > 0.3])

        # Kararlılık filtresi
        if plate_text:
            recent_plates.append(plate_text)
            most_common = Counter(recent_plates).most_common(1)[0]
            if most_common[1] >= 3:  # En az 3 kez arka arkaya aynıysa
                stable_plate = most_common[0]

        # Plaka çerçevesi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Yazı için konum ve arka plan kutusu
        text_position = (x1, y1 - 15)
        text_size, _ = cv2.getTextSize(stable_plate, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        text_w, text_h = text_size
        cv2.rectangle(frame, (text_position[0] - 5, text_position[1] - text_h - 10),
                      (text_position[0] + text_w + 5, text_position[1] + 10),
                      (0, 0, 0), -1)

        # Plaka yazısı
        cv2.putText(frame, stable_plate, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Ekranı sığdır
    max_width = 800
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # Görüntüyü göster
    cv2.imshow("Plate Detection", frame)

    # Çıkış: 'q' tuşu
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
