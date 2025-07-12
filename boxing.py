import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import pickle

SEQUENCE_LENGTH = 10  # Number of frames per sequence
FEATURE_SIZE = 32     # 16 landmarks x (x, y)

def normalize_landmarks(pose_vec):
    pose_vec = np.array(pose_vec).reshape(-1, 2)  # (16, 2)
    center = np.mean(pose_vec, axis=0)
    pose_vec -= center
    scale = np.std(pose_vec)
    if scale > 0:
        pose_vec /= scale
    return pose_vec.flatten()


print("[INFO] Loading dataset...")
df = pd.read_csv(r"boxing_var_balanced_dataset.csv")  # Replace with your CSV path
# Define the 16 landmark indices used for training
LANDMARK_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

print(f"[INFO] Dataset shape: {df.shape}")

# Separate features and labels
X_raw = df.iloc[:, :-1].values.astype(np.float32)
X_raw = np.array([normalize_landmarks(x) for x in X_raw])
y_raw = df.iloc[:, -1].values

print("[INFO] Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)

# Save the label encoder for inference later
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Debug class distribution
print("[DEBUG] Class distribution:", Counter(y_encoded))

# Create sequences
print("[INFO] Creating sequences...")
X_seq, y_seq = [], []

for i in range(len(X_raw) - SEQUENCE_LENGTH + 1):
    seq_x = X_raw[i:i + SEQUENCE_LENGTH]
    seq_y = y_encoded[i + SEQUENCE_LENGTH - 1]
    X_seq.append(seq_x)
    y_seq.append(seq_y)

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(f"[INFO] Final shape: X: {X_seq.shape}, y: {y_seq.shape}")

# Train/validation/test split
X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, stratify=y_seq, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Compute class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print("[DEBUG] Class weights:", class_weight_dict)

# Build the LSTM model
print("[INFO] Building model...")
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(len(classes), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train the model
print("[INFO] Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate the model
print("[INFO] Evaluating model...")
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))

print("[DEBUG] Class distribution:", Counter(y_encoded))


# Save the trained model
model.save("boxing_motion_classifier.h5")
print("[DONE] Model saved as 'boxing_motion_classifier.h5'")
