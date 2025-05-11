import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

model = YOLO("best.pt") 

# EasyOCR reader
reader = easyocr.Reader(['en'])

# Görüntüyü oku
frame = cv2.imread("image.jpg")

# YOLO ile plaka tespiti yap
results = model.predict(source=frame, conf=0.5, classes=0, verbose=False)
boxes = results[0].boxes.xyxy.cpu().numpy()

if len(boxes) == 0:
    print("Plaka bulunamadı.")

for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    plate_img = frame[y1:y2, x1:x2]

    if plate_img.size == 0:
        continue


    plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR işlemi
    ocr_result = reader.readtext(thresh)
    print("OCR Raw:", ocr_result)

    sorted_text = sorted(ocr_result, key=lambda x: x[0][0][0])
    plate_text = " ".join([text for (_, text, prob) in sorted_text if prob > 0.3])


    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Kırmızı çerçeve
    text_position = (x1, y1 - 15)


    text_size, _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    text_w, text_h = text_size
    cv2.rectangle(frame, (text_position[0] - 5, text_position[1] - text_h - 10),
                  (text_position[0] + text_w + 5, text_position[1] + 10),
                  (0, 0, 0), -1)

  
    cv2.putText(frame, plate_text, text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

r
max_width = 800
h, w = frame.shape[:2]
if w > max_width:
    scale = max_width / w
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))


frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


plt.imshow(frame_rgb)
plt.title("Plate Detection")
plt.axis('off')  
plt.show()
