from ultralytics import YOLO
import cv2
import os

# === CONFIG ===
model_path = "results/yolov8m/weights/best.pt"  # Path to your trained model
video_path = "firetest3.mp4"  # Path to your input video
output_path = "results/output_firetestm.mp4"  # Output video file
conf_threshold = 0.3  # Confidence threshold

# === Load YOLO model ===
model = YOLO(model_path)

# === Open video ===
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# === Set up video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === Inference loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction
    results = model.predict(frame, conf=conf_threshold, verbose=False)
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    out.write(annotated_frame)

    # Press Q to quit early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# === Cleanup ===
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Done! Annotated video saved to: {output_path}")