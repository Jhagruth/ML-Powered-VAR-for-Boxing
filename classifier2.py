import cv2
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
import time
import csv
import argparse
from ultralytics import YOLO

# ---------------------------
# Command-line arguments
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, help="Path to video file. Leave empty for webcam.")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video if args.video else 0)
print(f"[INFO] Using {'video file' if args.video else 'webcam'}...")

# ---------------------------
# Load LSTM model & label encoder
# ---------------------------
print("[INFO] Loading boxing motion classifier...")
model = tf.keras.models.load_model("boxing_motion_classifier.h5")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

class_names = {
    0: "Jab", 1: "Cross", 2: "Hook",
    3: "Uppercut", 4: "Block", 5: "Duck", 6: "Idle"
}

SEQUENCE_LENGTH = 10
FEATURE_SIZE = 32  # 16 keypoints * 2 (x, y)
sequences = {}

# ---------------------------
# Load YOLOv8 pose model
# ---------------------------
print("[INFO] Loading YOLOv8 pose model...")
yolo_model = YOLO("yolov8n-pose.pt")  # or yolov8m-pose.pt for better accuracy

# ---------------------------
# CSV Logging
# ---------------------------
csv_file = open("output_predictions.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp (min)", "PersonID", "Motion", "Confidence"])
start_time = time.time()
last_recorded_time = time.time()

# ---------------------------
# Utility: Normalize landmarks
# ---------------------------
def normalize_landmarks(pose_vec):
    pose_vec = np.array(pose_vec).reshape(-1, 2)
    center = np.mean(pose_vec, axis=0)
    pose_vec -= center
    scale = np.std(pose_vec)
    if scale > 0:
        pose_vec /= scale
    return pose_vec.flatten()

# ---------------------------
# Main loop
# ---------------------------
print("[INFO] Starting pose-based motion classification...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of stream.")
        break

    results = yolo_model(frame, verbose=False)[0]

    if results.keypoints is not None and results.keypoints.shape[0] > 0:
        # Sort detections by box confidence (descending)
        boxes = results.boxes
        confidences = boxes.conf.cpu().numpy()
        sorted_indices = np.argsort(confidences)[::-1]  # descending order

        keypoints_all = results.keypoints.data.cpu().numpy()
        boxes_all = boxes.xyxy.cpu().numpy()

        # Process top 2 people
        for i, idx in enumerate(sorted_indices[:2]):
            keypoints = keypoints_all[idx]
            box = boxes_all[idx]
            person_id = i
            x_min, y_min, x_max, y_max = box.astype(int)
            pose_vec = []

            for j in range(16):  # use only first 16 keypoints
                x, y, _ = keypoints[j]
                pose_vec.extend([x / frame.shape[1], y / frame.shape[0]])

            normed_vec = normalize_landmarks(pose_vec)

            if person_id not in sequences:
                sequences[person_id] = deque(maxlen=SEQUENCE_LENGTH)
            sequences[person_id].append(normed_vec)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

            # Predict if sequence is ready
            if len(sequences[person_id]) == SEQUENCE_LENGTH:
                input_seq = np.expand_dims(sequences[person_id], axis=0)
                preds = model.predict(input_seq, verbose=0)[0]
                class_id = int(np.argmax(preds))
                confidence = preds[class_id]
                label = class_names.get(class_id, f"Unknown_{class_id}")

                color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
                text = f"Person {person_id+1}: {label} ({confidence:.2f})"
                cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Save to CSV every 0.5s
                current_time = time.time()
                if current_time - last_recorded_time >= 0.5:
                    timestamp_minutes = round((current_time - start_time) / 60, 3)
                    csv_writer.writerow([timestamp_minutes, f"Person {person_id+1}", label, round(confidence, 3)])
                    last_recorded_time = current_time

    cv2.imshow("YOLO Pose Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------------------
# Cleanup
# ---------------------------
cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("[INFO] Saved predictions to 'output_predictions.csv'")
