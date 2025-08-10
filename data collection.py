import cv2
import os
import time

# Prompt user for the label
label = input("Enter the sign label (e.g., hello): ").strip().lower()
output_dir = f"dataset/{label}"
os.makedirs(output_dir, exist_ok=True)

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frame_count = 0
cooldown = 1  # seconds between captures

print(f"📷 Collecting images for: '{label}'")
print("👉 Press 'c' to capture an image | 'q' to quit")

last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    cv2.putText(frame, f"Label: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.putText(frame, f"Images saved: {frame_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("YOLOv8 Sign Dataset Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()

    if key == ord('c') and current_time - last_capture_time > cooldown:
        filename = os.path.join(output_dir, f"{label}_{frame_count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ Saved: {filename}")
        frame_count += 1
        last_capture_time = current_time

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n📁 Collection complete: {frame_count} images saved in '{output_dir}'")
