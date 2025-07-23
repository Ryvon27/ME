# Import required libraries
import cv2
from ultralytics import YOLO
import cvzone
import pandas as pd
import json

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
names = model.names

# Vertical line position
line_x = 480

# Previous center positions
hist = {}

# IN/OUT counters
in_count = 0
out_count = 0

# Open video file or webcam
cap = cv2.VideoCapture("vid1.mp4")  # Ganti ke 0 untuk webcam

# Check if video opened successfully
if not cap.isOpened():
    print("[ERROR] Video file not found or cannot be opened!")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Video ended.")
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue  # Skip every other frame

    frame = cv2.resize(frame, (1020, 600))

    # Detect and track persons (class 0)
    results = model.track(frame, persist=True, classes=[0])

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        
        for box, track_id, class_id in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = box
            c = names[class_id]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw boxes and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{c.upper()}', (x1, y1 - 10), scale=1, thickness=1,
                               colorT=(255, 255, 255), colorR=(0, 0, 255))
            cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y2 + 10), scale=1, thickness=1,
                               colorT=(255, 255, 255), colorR=(0, 255, 0))

            # Counting logic
            if track_id in hist:
                prev_cx, _ = hist[track_id]
                if prev_cx < line_x <= cx:
                    in_count += 1
                    print(f"[INFO] IN counted! ID: {track_id}")
                elif prev_cx > line_x >= cx:
                    out_count += 1
                    print(f"[INFO] OUT counted! ID: {track_id}")
            hist[track_id] = (cx, cy)

    # Display current total counts
    cvzone.putTextRect(frame, f'IN: {in_count}', (40, 60), scale=2, thickness=2,
                       colorT=(255, 255, 255), colorR=(0, 128, 0))
    cvzone.putTextRect(frame, f'OUT: {out_count}', (40, 100), scale=2, thickness=2,
                       colorT=(255, 255, 255), colorR=(0, 0, 255))
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 255, 255), 2)

    cv2.imshow("RGB", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        print("[INFO] ESC pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

# Save only final summary
summary = {
    "total_in": in_count,
    "total_out": out_count
}

# Save CSV & JSON (only summary, 1 row)
df = pd.DataFrame([summary])
df.to_csv("final_crossing_summary.csv", index=False)

with open("final_crossing_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Print summary
print("\n===== FINAL SUMMARY =====")
print(f"Total IN Count : {in_count}")
print(f"Total OUT Count: {out_count}")
print("=========================")
print("Data saved to final_crossing_summary.csv and final_crossing_summary.json")
