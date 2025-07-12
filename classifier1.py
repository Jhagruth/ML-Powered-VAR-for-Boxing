import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
import argparse
import time
import csv

# ---------------------------
# Argument parsing (webcam or video file)
# ---------------------------
parser = argparse.ArgumentParser(description="Real-time or video motion classification")
parser.add_argument("--video", type=str, default=None, help="Path to video file. Leave blank for webcam.")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video if args.video else 0)
print(f"[INFO] Using {'video file' if args.video else 'webcam'}...")

# ---------------------------
# Load model and label encoder
# ---------------------------
print("[INFO] Loading model and label encoder...")
model = tf.keras.models.load_model("boxing_motion_classifier.h5")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ---------------------------
# Define class names mapping
# ---------------------------
class_names = {
    0: "Jab",
    1: "Cross", 
    2: "Hook",
    3: "Uppercut",
    4: "Block",
    5: "Duck",
    6: "Idle"
}

# ---------------------------
# Mediapipe pose setup
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------------------
# Parameters
# ---------------------------
SEQUENCE_LENGTH = 10
FEATURE_SIZE = 32
LANDMARK_IDS = list(range(16))  # Use 0 to 15 for bounding box and features
sequence = deque(maxlen=SEQUENCE_LENGTH)

# ---------------------------
# Utility Functions
# ---------------------------
def normalize_landmarks(pose_vec):
    pose_vec = np.array(pose_vec).reshape(-1, 2)
    center = np.mean(pose_vec, axis=0)
    pose_vec -= center
    scale = np.std(pose_vec)
    if scale > 0:
        pose_vec /= scale
    return pose_vec.flatten()

def get_bounding_box(landmarks, frame_width, frame_height):
    x_coords = [landmarks[idx].x for idx in LANDMARK_IDS]
    y_coords = [landmarks[idx].y for idx in LANDMARK_IDS]
    x_min = int(min(x_coords) * frame_width)
    x_max = int(max(x_coords) * frame_width)
    y_min = int(min(y_coords) * frame_height)
    y_max = int(max(y_coords) * frame_height)
    return x_min, y_min, x_max, y_max

# ---------------------------
# Setup for CSV logging
# ---------------------------
csv_file = open("output_predictions.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp (min)", "Motion", "Confidence"])
start_time = time.time()
last_recorded_time = time.time()

# ---------------------------
# Main loop
# ---------------------------
print("[INFO] Starting classification...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of stream or camera not found.")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        pose_vec = []

        for idx in LANDMARK_IDS:
            lm = landmarks[idx]
            pose_vec.extend([lm.x, lm.y])

        # Normalize
        normed_vec = normalize_landmarks(pose_vec)

        # Add to sequence
        sequence.append(normed_vec)

        # Get frame dimensions
        frame_height, frame_width, _ = frame.shape

        # Draw bounding box
        x_min, y_min, x_max, y_max = get_bounding_box(landmarks, frame_width, frame_height)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

        # Predict if sequence ready
        if len(sequence) == SEQUENCE_LENGTH:
            input_seq = np.expand_dims(sequence, axis=0)  # Shape: (1, 10, 32)
            preds = model.predict(input_seq, verbose=0)[0]
            class_id = int(np.argmax(preds))
            
            label = class_names.get(class_id, f"Unknown_{class_id}")
            confidence = preds[class_id]

            # Display label near bounding box
            text = f"{label} ({confidence:.2f})"
            color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
            cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Save to CSV every 0.5 seconds
            current_time = time.time()
            if current_time - last_recorded_time >= 0.5:
                timestamp_minutes = round((current_time - start_time) / 60, 3)
                csv_writer.writerow([timestamp_minutes, label, round(confidence, 3)])
                last_recorded_time = current_time

    cv2.imshow("Motion Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("[INFO] CSV file saved as 'output_predictions.csv'")
