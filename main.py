from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO("runs/detect/train_signs2/weights/best.pt")  # Adjust path if needed

# Open webcam and set resolution similar to training (680x480 ~ 4:3)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise IOError("❌ Cannot open webcam")

print("🖐 Start signing! Press 'q' to quit.\n")

transcript = []
last_label = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and crop center
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    # Optional: center crop to focus on hands


    # Match training image size
    results = model.predict(source=frame, imgsz=480, conf=0.70, verbose=False)
    annotated_frame = frame.copy()
    detected_signs = []

    for r in results:
        boxes = r.boxes
        print(f"[DEBUG] {len(boxes)} detections")
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = box.conf[0].item()
            detected_signs.append(label)
            print(f"Detected: {label} (conf: {conf:.2f})")

            # Draw bounding box
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Append to transcript if not duplicate
    if detected_signs:
        if detected_signs[0] != last_label:
            transcript.append(detected_signs[0])
            print("✍ Added to transcript:", detected_signs[0])
            last_label = detected_signs[0]
    else:
        cv2.putText(
            annotated_frame, "No sign detected", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

    # Display transcript on screen
    cv2.putText(
        annotated_frame,
        "Transcript: " + ' '.join(transcript[-6:]),
        (10, annotated_frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.imshow("YOLOv8 Sign Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Final result
print("\n📝 Final Transcript:")
print(' '.join(transcript))
