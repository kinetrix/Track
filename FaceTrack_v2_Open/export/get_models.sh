#!/usr/bin/env bash
# Download the models that are used as published (OpenCV Zoo, via GitHub LFS) into ../models.
set -euo pipefail
cd "$(dirname "$0")/../models"
Z=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models
curl -fL -o face_detection_yunet_2023mar.onnx $Z/face_detection_yunet/face_detection_yunet_2023mar.onnx            # MIT
curl -fL -o palm_detection_mediapipe_2023feb.onnx $Z/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx   # Apache-2.0
curl -fL -o handpose_estimation_mediapipe_2023feb.onnx $Z/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx  # Apache-2.0
ls -la *.onnx
