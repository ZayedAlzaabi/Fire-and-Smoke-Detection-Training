from ultralytics import YOLO
import cv2
import os
import time

# === CONFIG ===
model_path = "results/yolo12n/weights/best.pt"  # Path to your trained model
video_path = "firetest3.mp4"  # Path to your input video
output_path = "results/output_firetest2.mp4"  # Output video file
conf_threshold = 0.3  # Confidence threshold

# === Load YOLO model ===
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")
model = YOLO(model_path)

# === Open video ===
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found at {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"Could not open video at {video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# === Set up video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    raise RuntimeError("Could not initialize video writer")

# === Inference loop ===
frame_count = 0
start_time = time.time()
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run prediction
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()

        # Write frame
        out.write(annotated_frame)
        frame_count += 1

        # Show progress
        if frame_count % 10 == 0:  # Update every 10 frames
            elapsed_time = time.time() - start_time
            fps_processing = frame_count / elapsed_time
            progress = (frame_count / total_frames) * 100
            print(f"\rProcessing: {progress:.1f}% ({frame_count}/{total_frames}) - {fps_processing:.1f} fps", end="")

except Exception as e:
    print(f"\n❌ Error during processing: {str(e)}")
finally:
    # === Cleanup ===
    cap.release()
    out.release()
    print(f"\n✅ Processed {frame_count} frames")
    print(f"✅ Annotated video saved to: {output_path}")