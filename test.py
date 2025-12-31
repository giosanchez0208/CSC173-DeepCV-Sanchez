import cv2
from ultralytics import YOLO

model = YOLO("models/custom_ocr.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, stream=True, verbose=False, conf=0.4)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Formatting
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            label = f"{cls_name} {conf:.1f}"
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Draw centered small text
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thickness = 0.4, 1
            (w, h), _ = cv2.getTextSize(label, font, scale, thickness)
            cv2.putText(frame, label, (cx - w//2, cy + h//2), font, scale, (255, 255, 255), thickness)

    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()