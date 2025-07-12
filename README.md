# ML-Powered VAR for Boxing

> A computer vision-powered system that brings **Video Assistant Referee (VAR)** capabilities to boxing by detecting and classifying punches in real-time or pre-recorded footage.

---

## 🚀 Overview

This project leverages **deep learning**, **pose estimation**, and optionally **YOLO** or **MediaPipe**, to detect and classify boxing punches in both **live video** and **recorded matches**. The system is designed to assist coaches, referees, and analysts by providing accurate insights on punch types and patterns.

---

## 🎯 Features

- ✅ Real-time punch classification using webcam/video
- ✅ EfficientNet + LSTM models trained on annotated Olympic boxing datasets
- ✅ Pose landmark-based feature extraction
- ✅ Support for both MediaPipe and YOLOv8 pose estimation
- ✅ Live skeleton overlay and smoothed predictions
- ✅ Works with pre-recorded `.mp4` or `.mov` videos
- ✅ Modular architecture (easily swappable models/components)

---

## 🧠 Model Architecture

- **Feature Extractor**: Pose-based keypoints from MediaPipe or YOLO
- **Input**: 10-frame sequences × 14 features (7 keypoints)
- **Classifier**: LSTM + Dense layers
- **Output**: Punch type (e.g., Jab, Cross, Hook, Uppercut, Idle)

---

## 🗃️ Dataset

- **Source**: [Olympic Boxing Punch Classification Dataset](https://www.kaggle.com/datasets/piotrstefaskiue/olympic-boxing-punch-classification-video-dataset), Custom Synthetic Data
- **Preprocessing**:
  - Frame extraction & annotation
  - Pose landmark detection
  - Sequence creation
  - Label smoothing & class balancing

---

## 🛠️ Setup

### 1. Clone the Repo

```bash
git clone git@github.com:Jhagruth/ML-Powered-VAR-for-Boxing.git
cd ML-Powered-VAR-for-Boxing
